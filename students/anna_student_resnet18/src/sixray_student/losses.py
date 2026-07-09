"""
Loss functions for Anna's ResNet18 YOLO-style student detector.

The student predicts:

    predictions: [B, K * (1 + 4 + C), S, S]

where:
- B = batch size
- K = number of box slots per grid cell
- C = number of classes
- S = grid size

For each box slot, the prediction layout is:

    [objectness_logit, tx, ty, w, h, class_logits...]

Loss components:
- objectness: BCEWithLogitsLoss over all grid cells and slots
- bbox: Smooth L1 loss only on positive slots
- class: cross entropy only on positive slots
"""

import torch
import torch.nn.functional as F

from sixray_student.config import (
    NUM_CLASSES,
    NUM_BOXES,
    BOX_LOSS_WEIGHT,
    MAX_OBJECTNESS_POS_WEIGHT,
)


def split_predictions(
    predictions,
    num_classes=NUM_CLASSES,
    num_boxes=NUM_BOXES,
):
    """
    Split raw model predictions into objectness, bbox, and class tensors.

    Args:
        predictions:
            Tensor with shape:
                [B, K * (1 + 4 + C), S, S]

    Returns:
        objectness_logits:
            [B, K, 1, S, S]

        bbox_raw:
            [B, K, 4, S, S]

        class_logits:
            [B, K, C, S, S]
    """

    batch_size, channels, height, width = predictions.shape

    values_per_box = 1 + 4 + num_classes
    expected_channels = num_boxes * values_per_box

    if channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels for "
            f"num_boxes={num_boxes}, num_classes={num_classes}, "
            f"but got {channels}."
        )

    predictions = predictions.view(
        batch_size,
        num_boxes,
        values_per_box,
        height,
        width,
    )

    objectness_logits = predictions[:, :, 0:1, :, :]
    bbox_raw = predictions[:, :, 1:5, :, :]
    class_logits = predictions[:, :, 5:, :, :]

    return objectness_logits, bbox_raw, class_logits


def objectness_loss(
    objectness_logits,
    objectness_target,
    max_pos_weight=MAX_OBJECTNESS_POS_WEIGHT,
):
    """
    Compute objectness loss.

    Most grid cells are empty, so positives are rare.
    We use a dynamic positive weight:

        positive_weight = num_negative / num_positive

    but cap it to avoid unstable gradients.

    Args:
        objectness_logits:
            [B, K, 1, S, S]

        objectness_target:
            [B, K, 1, S, S], values 0 or 1
    """

    positive_count = objectness_target.sum()
    total_count = objectness_target.numel()
    negative_count = total_count - positive_count

    if positive_count.item() == 0:
        positive_weight = torch.tensor(
            1.0,
            device=objectness_logits.device,
            dtype=objectness_logits.dtype,
        )
    else:
        positive_weight = negative_count / positive_count
        positive_weight = positive_weight.clamp(max=max_pos_weight)

    loss = F.binary_cross_entropy_with_logits(
        objectness_logits,
        objectness_target,
        pos_weight=positive_weight,
    )

    return loss


def box_loss(bbox_raw, bbox_target, positive_mask):
    """
    Compute bbox regression loss only for positive slots.

    Args:
        bbox_raw:
            [B, K, 4, S, S]

        bbox_target:
            [B, K, 4, S, S]

        positive_mask:
            [B, K, 1, S, S]

    Bbox format:
        [tx, ty, w, h]

    where:
        tx, ty are local cell offsets
        w, h are normalized by image size

    The model outputs raw bbox values.
    We apply sigmoid so predictions are constrained to [0, 1].
    """

    bbox_pred = torch.sigmoid(bbox_raw)

    loss_per_coord = F.smooth_l1_loss(
        bbox_pred,
        bbox_target,
        reduction="none",
    )

    loss_per_coord = loss_per_coord * positive_mask

    num_positive_slots = positive_mask.sum()

    if num_positive_slots.item() == 0:
        return torch.tensor(
            0.0,
            device=bbox_raw.device,
            dtype=bbox_raw.dtype,
        )

    num_bbox_coordinates = bbox_raw.shape[2]

    loss = loss_per_coord.sum() / (
        num_positive_slots * num_bbox_coordinates
    )

    return loss


def class_loss(class_logits, target_classes):
    """
    Compute classification loss only for positive slots.

    Args:
        class_logits:
            [B, K, C, S, S]

        target_classes:
            [B, K, S, S]

    target_classes values:
        0, 1, 2, ... = real class id
        -1 = ignore / no object
    """

    valid_count = (target_classes >= 0).sum()

    if valid_count.item() == 0:
        return torch.tensor(
            0.0,
            device=class_logits.device,
            dtype=class_logits.dtype,
        )

    batch_size, num_boxes, num_classes, grid_h, grid_w = class_logits.shape

    class_logits = class_logits.reshape(
        batch_size * num_boxes,
        num_classes,
        grid_h,
        grid_w,
    )

    target_classes = target_classes.reshape(
        batch_size * num_boxes,
        grid_h,
        grid_w,
    )

    loss = F.cross_entropy(
        class_logits,
        target_classes,
        ignore_index=-1,
    )

    return loss


def _get_class_targets(encoded_targets):
    """
    Read class targets from encoded target dictionary.

    The notebook originally used:
        encoded_targets["classes"]

    The refactored encoder uses:
        encoded_targets["class_targets"]

    This helper accepts both names.
    """

    if "class_targets" in encoded_targets:
        return encoded_targets["class_targets"]

    if "classes" in encoded_targets:
        return encoded_targets["classes"]

    raise KeyError(
        "encoded_targets must contain either 'class_targets' or 'classes'."
    )


def yolo_loss(
    predictions,
    encoded_targets,
    box_loss_weight=BOX_LOSS_WEIGHT,
    max_objectness_pos_weight=MAX_OBJECTNESS_POS_WEIGHT,
):
    """
    Compute total YOLO-style student loss.

    Total loss:

        objectness_loss
        + box_loss_weight * bbox_loss
        + class_loss

    Returns:
        total_loss:
            PyTorch scalar used for backpropagation.

        loss_dict:
            Python floats for logging.
    """

    objectness_logits, bbox_raw, class_logits = split_predictions(predictions)

    target_classes = _get_class_targets(encoded_targets)

    obj_loss = objectness_loss(
        objectness_logits=objectness_logits,
        objectness_target=encoded_targets["objectness"],
        max_pos_weight=max_objectness_pos_weight,
    )

    b_loss = box_loss(
        bbox_raw=bbox_raw,
        bbox_target=encoded_targets["bbox"],
        positive_mask=encoded_targets["positive_mask"],
    )

    c_loss = class_loss(
        class_logits=class_logits,
        target_classes=target_classes,
    )

    total_loss = obj_loss + box_loss_weight * b_loss + c_loss

    loss_dict = {
        "objectness_loss": float(obj_loss.detach().cpu()),
        "bbox_loss": float(b_loss.detach().cpu()),
        "class_loss": float(c_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }

    return total_loss, loss_dict


def check_loss_on_dummy_batch():
    """
    Small CPU sanity check for the loss code.

    This does not test model quality.
    It only checks that shapes are compatible and the loss runs.
    """

    batch_size = 2
    grid_size = 20
    num_boxes = NUM_BOXES
    num_classes = NUM_CLASSES

    channels = num_boxes * (1 + 4 + num_classes)

    predictions = torch.randn(
        batch_size,
        channels,
        grid_size,
        grid_size,
    )

    objectness = torch.zeros(
        batch_size,
        num_boxes,
        1,
        grid_size,
        grid_size,
    )

    bbox = torch.zeros(
        batch_size,
        num_boxes,
        4,
        grid_size,
        grid_size,
    )

    class_targets = torch.full(
        (batch_size, num_boxes, grid_size, grid_size),
        fill_value=-1,
        dtype=torch.long,
    )

    positive_mask = torch.zeros(
        batch_size,
        num_boxes,
        1,
        grid_size,
        grid_size,
        dtype=torch.bool,
    )

    objectness[0, 0, 0, 3, 4] = 1.0
    bbox[0, 0, :, 3, 4] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    class_targets[0, 0, 3, 4] = 2
    positive_mask[0, 0, 0, 3, 4] = True

    encoded_targets = {
        "objectness": objectness,
        "bbox": bbox,
        "class_targets": class_targets,
        "positive_mask": positive_mask,
    }

    total_loss, loss_dict = yolo_loss(predictions, encoded_targets)

    print("Total loss:", float(total_loss.detach().cpu()))
    print(loss_dict)

    return total_loss, loss_dict