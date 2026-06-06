import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert


def _to_metric_target(labels: list[dict], device: torch.device) -> list[dict]:
    """Convert processor labels to torchmetrics format (absolute xyxy boxes)."""
    targets = []
    for label in labels:
        h, w = label["orig_size"].tolist()
        boxes = label["boxes"].to(device)
        abs_cxcywh = boxes.clone()
        abs_cxcywh[:, 0] *= w
        abs_cxcywh[:, 1] *= h
        abs_cxcywh[:, 2] *= w
        abs_cxcywh[:, 3] *= h
        abs_xyxy = box_convert(abs_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
        targets.append({
            "boxes": abs_xyxy,
            "labels": label["class_labels"].to(device),
        })
    return targets


def _to_metric_preds(
    results: list[dict],
    device: torch.device,
    score_threshold: float = 0.0,
) -> list[dict]:
    preds = []
    for result in results:
        scores = result["scores"]
        keep = scores >= score_threshold
        preds.append({
            "boxes": result["boxes"][keep].to(device),
            "scores": scores[keep].to(device),
            "labels": result["labels"][keep].to(device),
        })
    return preds


@torch.no_grad()
def evaluate_detection(
    model,
    processor,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.0,
) -> dict:
    """Run validation and return COCO-style mAP from torchmetrics."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)
        target_sizes = torch.stack([label["orig_size"] for label in labels])
        results = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        preds = _to_metric_preds(results, device, score_threshold)
        targets = _to_metric_target(labels, device)
        metric.update(preds, targets)

    computed = metric.compute()
    return {
        "map": float(computed["map"].item()),
        "map_50": float(computed["map_50"].item()),
        "map_75": float(computed["map_75"].item()),
    }
