"""
Smoke test for Anna's ResNet18 YOLO-style student detector.

This script checks that the refactored modules work together without needing
the real SIXray-D dataset.

Run from repository root:

    python students/anna_student_resnet18/scripts/smoke_test_student.py
"""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from sixray_student.config import (
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
)

from sixray_student.model import (
    build_student_model,
    count_total_parameters,
    count_trainable_parameters,
)

from sixray_student.target_encoder import encode_targets_yolo
from sixray_student.losses import yolo_loss
from sixray_student.metrics import decode_predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("IMAGE_SIZE:", IMAGE_SIZE)
    print("GRID_SIZE:", GRID_SIZE)
    print("NUM_BOXES:", NUM_BOXES)
    print("NUM_CLASSES:", NUM_CLASSES)

    print("\nBuilding model...")

    # Important:
    # pretrained_backbone=False avoids downloading ImageNet weights during a local smoke test.
    model = build_student_model(pretrained_backbone=False)
    model = model.to(device)
    model.eval()

    print("Total parameters:", count_total_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))

    print("\nRunning dummy forward pass...")

    batch_size = 2

    images = torch.rand(
        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    with torch.no_grad():
        predictions = model(images)

    expected_channels = NUM_BOXES * (1 + 4 + NUM_CLASSES)
    expected_shape = (
        batch_size,
        expected_channels,
        GRID_SIZE,
        GRID_SIZE,
    )

    print("Predictions shape:", tuple(predictions.shape))
    print("Expected shape:", expected_shape)

    if tuple(predictions.shape) != expected_shape:
        raise RuntimeError(
            f"Wrong prediction shape: got {tuple(predictions.shape)}, "
            f"expected {expected_shape}"
        )

    print("\nEncoding dummy targets...")

    targets = [
        {
            "boxes": torch.tensor(
                [
                    [50.0, 60.0, 160.0, 190.0],
                    [300.0, 320.0, 420.0, 460.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
            "labels": torch.tensor(
                [0, 3],
                dtype=torch.long,
                device=device,
            ),
            "image_id": torch.tensor([1], dtype=torch.long),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "image_id": torch.tensor([2], dtype=torch.long),
        },
    ]

    encoded = encode_targets_yolo(
        targets=targets,
        image_size=IMAGE_SIZE,
        grid_size=GRID_SIZE,
        num_boxes=NUM_BOXES,
        device=device,
    )

    print("Objectness target:", tuple(encoded["objectness"].shape))
    print("Bbox target:", tuple(encoded["bbox"].shape))
    print("Class target:", tuple(encoded["class_targets"].shape))
    print("Positive mask:", tuple(encoded["positive_mask"].shape))
    print("Assigned objects:", encoded["num_assigned"])
    print("Skipped objects:", encoded["num_skipped"])

    if encoded["num_assigned"] != 2:
        raise RuntimeError(
            f"Expected 2 assigned objects, got {encoded['num_assigned']}"
        )

    print("\nComputing dummy loss...")

    loss, loss_dict = yolo_loss(
        predictions=predictions,
        encoded_targets=encoded,
    )

    print("Loss:", float(loss.detach().cpu()))
    print("Loss dict:", loss_dict)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    print("\nDecoding dummy predictions...")

    decoded = decode_predictions(
        predictions=predictions,
        confidence_threshold=0.05,
        nms_iou_threshold=0.50,
        max_detections=100,
    )

    print("Decoded images:", len(decoded))

    for image_index, pred in enumerate(decoded):
        print(
            f"Image {image_index}: "
            f"boxes={tuple(pred['boxes'].shape)}, "
            f"scores={tuple(pred['scores'].shape)}, "
            f"labels={tuple(pred['labels'].shape)}"
        )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()