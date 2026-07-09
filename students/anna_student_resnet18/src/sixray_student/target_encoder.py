"""

The student predicts a fixed grid:

    [B, NUM_BOXES * (1 + 4 + NUM_CLASSES), GRID_SIZE, GRID_SIZE]

For each image, each ground-truth box is assigned to the grid cell containing
its center.

Bbox encoding:
- tx, ty: local offsets inside the assigned grid cell, in [0, 1]
- w, h: normalized width and height relative to the full image size

Target tensors:
- objectness: [B, K, 1, S, S]
- bbox:       [B, K, 4, S, S]
- class:      [B, K, S, S]
- mask:       [B, K, 1, S, S]

where:
- B = batch size
- K = number of box slots per cell
- S = grid size
"""

from collections import Counter

import torch

from sixray_student.config import (
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_BOXES,
)


def _safe_cell_index(value, grid_size):
    """
    Convert a continuous grid coordinate to a valid integer cell index.

    Example:
        value = 19.999 for a 20x20 grid -> cell 19

    This also protects against rare numerical edge cases where the center lies
    exactly on the image boundary.
    """

    cell_index = int(value)

    if cell_index < 0:
        cell_index = 0

    if cell_index >= grid_size:
        cell_index = grid_size - 1

    return cell_index


def _find_free_slot(objectness_targets, batch_index, grid_y, grid_x, num_boxes):
    """
    Find an empty box slot in one grid cell.

    Returns:
        slot index if an empty slot exists.
        None if all slots are already occupied.
    """

    for box_slot in range(num_boxes):
        is_occupied = objectness_targets[
            batch_index,
            box_slot,
            0,
            grid_y,
            grid_x,
        ] > 0

        if not is_occupied:
            return box_slot

    return None


def encode_targets_yolo(
    targets,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    num_boxes=NUM_BOXES,
    device=None,
):
    """
    Encode a batch of detection targets into YOLO-style grid targets.

    Args:
        targets:
            List of dictionaries. Each dictionary should contain:
                target["boxes"]:  Tensor [N, 4] in absolute xyxy pixel format.
                target["labels"]: Tensor [N] with integer class ids.

        image_size:
            Student input image size. The current student uses 640.

        grid_size:
            Number of grid cells per spatial dimension. The current student uses 20.

        num_boxes:
            Number of box slots per grid cell. The current student uses 2.

        device:
            Device where target tensors should be created. If None, the device is
            inferred from the first non-empty target boxes tensor.

    Returns:
        Dictionary with:
            objectness:    [B, K, 1, S, S]
            bbox:          [B, K, 4, S, S]
            class_targets: [B, K, S, S]
            positive_mask: [B, K, 1, S, S]
            num_assigned:  int
            num_skipped:   int
    """

    batch_size = len(targets)

    if device is None:
        device = torch.device("cpu")

        for target in targets:
            boxes = target.get("boxes")

            if boxes is not None and len(boxes) > 0:
                device = boxes.device
                break

    objectness_targets = torch.zeros(
        (batch_size, num_boxes, 1, grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    bbox_targets = torch.zeros(
        (batch_size, num_boxes, 4, grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    class_targets = torch.full(
        (batch_size, num_boxes, grid_size, grid_size),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )

    positive_mask = torch.zeros(
        (batch_size, num_boxes, 1, grid_size, grid_size),
        dtype=torch.bool,
        device=device,
    )

    num_assigned = 0
    num_skipped = 0

    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        labels = target["labels"].to(device=device, dtype=torch.long)

        if boxes.numel() == 0:
            continue

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                num_skipped += 1
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            center_x_norm = center_x / image_size
            center_y_norm = center_y / image_size
            width_norm = box_width / image_size
            height_norm = box_height / image_size

            # Convert normalized center to continuous grid coordinates.
            grid_x_float = center_x_norm * grid_size
            grid_y_float = center_y_norm * grid_size

            grid_x = _safe_cell_index(grid_x_float, grid_size)
            grid_y = _safe_cell_index(grid_y_float, grid_size)

            # Local offset inside the selected grid cell.
            tx = grid_x_float - grid_x
            ty = grid_y_float - grid_y

            # Numerical safety.
            tx = max(0.0, min(float(tx), 1.0))
            ty = max(0.0, min(float(ty), 1.0))
            width_norm = max(0.0, min(float(width_norm), 1.0))
            height_norm = max(0.0, min(float(height_norm), 1.0))

            box_slot = _find_free_slot(
                objectness_targets=objectness_targets,
                batch_index=batch_index,
                grid_y=grid_y,
                grid_x=grid_x,
                num_boxes=num_boxes,
            )

            if box_slot is None:
                # More than num_boxes objects have centers in the same cell.
                # We skip the extra object because the student has no free slot.
                num_skipped += 1
                continue

            objectness_targets[
                batch_index,
                box_slot,
                0,
                grid_y,
                grid_x,
            ] = 1.0

            bbox_targets[
                batch_index,
                box_slot,
                :,
                grid_y,
                grid_x,
            ] = torch.tensor(
                [tx, ty, width_norm, height_norm],
                dtype=torch.float32,
                device=device,
            )

            class_targets[
                batch_index,
                box_slot,
                grid_y,
                grid_x,
            ] = label

            positive_mask[
                batch_index,
                box_slot,
                0,
                grid_y,
                grid_x,
            ] = True

            num_assigned += 1

    encoded_targets = {
        "objectness": objectness_targets,
        "bbox": bbox_targets,
        "class_targets": class_targets,
        "positive_mask": positive_mask,
        "num_assigned": num_assigned,
        "num_skipped": num_skipped,
    }

    return encoded_targets


def count_grid_collisions(
    targets,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
):
    """
    Count how many ground-truth boxes fall into already occupied grid cells.

    This ignores NUM_BOXES and counts collisions at the cell level.

    Example:
        If two objects have centers in the same grid cell, this counts as one
        collision, even if NUM_BOXES=2 can still represent both objects.

    Returns:
        Dictionary with collision statistics.
    """

    total_boxes = 0
    occupied_cells = 0
    collision_boxes = 0
    images_with_collisions = 0

    for target in targets:
        boxes = target["boxes"]

        if boxes.numel() == 0:
            continue

        cell_counter = Counter()

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            grid_x_float = (center_x / image_size) * grid_size
            grid_y_float = (center_y / image_size) * grid_size

            grid_x = _safe_cell_index(grid_x_float, grid_size)
            grid_y = _safe_cell_index(grid_y_float, grid_size)

            cell_counter[(grid_y, grid_x)] += 1
            total_boxes += 1

        occupied_cells += len(cell_counter)

        image_collision_boxes = sum(
            count - 1
            for count in cell_counter.values()
            if count > 1
        )

        if image_collision_boxes > 0:
            images_with_collisions += 1
            collision_boxes += image_collision_boxes

    collision_rate = 0.0

    if total_boxes > 0:
        collision_rate = collision_boxes / total_boxes

    return {
        "total_boxes": total_boxes,
        "occupied_cells": occupied_cells,
        "collision_boxes": collision_boxes,
        "images_with_collisions": images_with_collisions,
        "collision_rate": collision_rate,
    }


def print_grid_collision_stats(stats):
    """
    Pretty-print grid collision statistics.
    """

    print("Grid collision statistics")
    print("-------------------------")
    print(f"Total boxes:             {stats['total_boxes']}")
    print(f"Occupied cells:          {stats['occupied_cells']}")
    print(f"Collision boxes:         {stats['collision_boxes']}")
    print(f"Images with collisions:  {stats['images_with_collisions']}")
    print(f"Collision rate:          {stats['collision_rate']:.4f}")