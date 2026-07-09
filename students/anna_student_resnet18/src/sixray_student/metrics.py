"""
This module contains:
- conversion from model bbox outputs to xyxy boxes
- prediction decoding with confidence filtering and NMS
- mAP evaluation using torchmetrics
- readable printing of detection metrics

It does not contain:
- training loop
- checkpoint saving/loading
- W&B logging
"""

import torch
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import batched_nms

from sixray_student.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    MAP_EVAL_CONFIDENCE_THRESHOLD,
    MAP_EVAL_NMS_IOU_THRESHOLD,
    MAP_EVAL_MAX_DETECTIONS,
)

from sixray_student.losses import split_predictions


def local_boxes_to_xyxy(bbox_raw, image_size=IMAGE_SIZE):
    """
    Convert raw local YOLO-style bbox predictions to absolute xyxy boxes.

    Args:
        bbox_raw:
            Tensor with shape [K, 4, S, S]

            K = number of box slots per grid cell
            S = grid size

            Raw bbox channels:
                tx, ty, w, h

    Encoding:
        tx, ty are local offsets inside the grid cell.
        w, h are normalized global image width and height.

    Returns:
        boxes:
            Tensor with shape [K * S * S, 4] in absolute xyxy pixel format.
    """

    bbox = torch.sigmoid(bbox_raw)

    tx = bbox[:, 0, :, :]
    ty = bbox[:, 1, :, :]
    w = bbox[:, 2, :, :]
    h = bbox[:, 3, :, :]

    _, grid_h, grid_w = tx.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=bbox.device),
        torch.arange(grid_w, device=bbox.device),
        indexing="ij",
    )

    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    cx = (grid_x + tx) / grid_w
    cy = (grid_y + ty) / grid_h

    x1 = (cx - w / 2.0) * image_size
    y1 = (cy - h / 2.0) * image_size
    x2 = (cx + w / 2.0) * image_size
    y2 = (cy + h / 2.0) * image_size

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes = boxes.reshape(-1, 4)

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size)

    return boxes


@torch.no_grad()
def decode_predictions(
    predictions,
    confidence_threshold=MAP_EVAL_CONFIDENCE_THRESHOLD,
    nms_iou_threshold=MAP_EVAL_NMS_IOU_THRESHOLD,
    max_detections=MAP_EVAL_MAX_DETECTIONS,
):
    """
    Decode raw model predictions into detection dictionaries.

    Args:
        predictions:
            Raw model output with shape [B, K * (1 + 4 + C), S, S]

        confidence_threshold:
            Minimum final detection score.

        nms_iou_threshold:
            IoU threshold for class-aware NMS.

        max_detections:
            Maximum number of detections kept per image.

    Returns:
        List of dictionaries compatible with torchmetrics MeanAveragePrecision:

        [
            {
                "boxes":  Tensor [N, 4],
                "scores": Tensor [N],
                "labels": Tensor [N],
            },
            ...
        ]
    """

    objectness_logits, bbox_raw, class_logits = split_predictions(predictions)

    batch_size = predictions.shape[0]
    decoded = []

    for image_index in range(batch_size):
        objectness = torch.sigmoid(objectness_logits[image_index]).squeeze(1)

        class_probs = F.softmax(class_logits[image_index], dim=1)

        scores_per_class = objectness.unsqueeze(1) * class_probs
        scores, labels = scores_per_class.max(dim=1)

        boxes = local_boxes_to_xyxy(bbox_raw[image_index])

        scores = scores.reshape(-1)
        labels = labels.reshape(-1).long()

        keep = scores >= confidence_threshold

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] > 0:
            keep = batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=labels,
                iou_threshold=nms_iou_threshold,
            )

            keep = keep[:max_detections]

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        decoded.append(
            {
                "boxes": boxes.detach().cpu(),
                "scores": scores.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )

    return decoded


def metrics_to_dict(metrics):
    """
    Convert torchmetrics output to plain Python values.

    This makes metrics easier to:
    - print
    - save as JSON
    - log to W&B
    """

    result = {}

    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()

            if value.numel() == 1:
                result[key] = float(value.item())
            else:
                result[key] = value.tolist()
        else:
            result[key] = value

    return result


def make_ground_truth_for_map(targets):
    """
    Convert batch targets to torchmetrics ground-truth format.

    Args:
        targets:
            List of dictionaries from the detection dataloader.

    Returns:
        List of dictionaries:
            [
                {
                    "boxes": Tensor [N, 4],
                    "labels": Tensor [N],
                },
                ...
            ]
    """

    ground_truth = []

    for target in targets:
        ground_truth.append(
            {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu().long(),
            }
        )

    return ground_truth


@torch.no_grad()
def evaluate_map(
    model,
    data_loader,
    device,
    confidence_threshold=MAP_EVAL_CONFIDENCE_THRESHOLD,
    nms_iou_threshold=MAP_EVAL_NMS_IOU_THRESHOLD,
    max_detections=MAP_EVAL_MAX_DETECTIONS,
    max_batches=None,
    use_amp=None,
):
    """
    Evaluate detection quality with mAP.

    Args:
        model:
            Student detection model.

        data_loader:
            Validation or test DataLoader.

        device:
            torch.device("cuda") or torch.device("cpu")

        confidence_threshold:
            Score threshold used during decoding.

        nms_iou_threshold:
            Class-aware NMS IoU threshold.

        max_detections:
            Maximum detections per image.

        max_batches:
            If not None, evaluate only this many batches.
            Useful for quick debugging.

        use_amp:
            If None, AMP is enabled only on CUDA.
            If True/False, use the explicit value.

    Returns:
        Dictionary with mAP metrics.
    """

    model.eval()

    if use_amp is None:
        use_amp = device.type == "cuda"

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    for batch_idx, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                predictions = model(images)
        else:
            predictions = model(images)

        preds = decode_predictions(
            predictions=predictions,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

        ground_truth = make_ground_truth_for_map(targets)

        metric.update(preds, ground_truth)

        if (batch_idx + 1) % 100 == 0:
            print(f"evaluated batches: {batch_idx + 1} / {len(data_loader)}")

    metrics = metric.compute()

    return metrics_to_dict(metrics)


def print_detection_metrics(
    metrics,
    title="Detection metrics",
    class_names=CLASS_NAMES,
):
    """
    Print detection metrics in a readable format.
    """

    print("\n" + title)
    print("-" * len(title))

    print("mAP:    ", round(metrics["map"], 4))
    print("mAP@50: ", round(metrics["map_50"], 4))
    print("mAP@75: ", round(metrics["map_75"], 4))

    print("\nPer-class mAP:")

    class_maps = metrics.get("map_per_class", [])

    for class_id, class_name in enumerate(class_names):
        if class_id >= len(class_maps):
            print(f"{class_name}: not available")
            continue

        value = class_maps[class_id]

        if value < 0:
            print(f"{class_name}: not available")
        else:
            print(f"{class_name}: {round(value, 4)}")


def check_decode_on_dummy_predictions():
    """
    Small sanity check for prediction decoding.

    This checks only tensor shapes and decoding mechanics.
    It does not check model quality.
    """

    from sixray_student.config import GRID_SIZE, NUM_BOXES, NUM_CLASSES

    batch_size = 2
    channels = NUM_BOXES * (1 + 4 + NUM_CLASSES)

    predictions = torch.randn(
        batch_size,
        channels,
        GRID_SIZE,
        GRID_SIZE,
    )

    decoded = decode_predictions(
        predictions,
        confidence_threshold=0.05,
        nms_iou_threshold=0.50,
        max_detections=100,
    )

    print("Number of images:", len(decoded))

    for image_index, pred in enumerate(decoded):
        print(f"Image {image_index}")
        print("boxes:", pred["boxes"].shape)
        print("scores:", pred["scores"].shape)
        print("labels:", pred["labels"].shape)

    return decoded