"""
Evaluate Anna's ResNet18 YOLO-style student detector.

Run from repository root:

    python students/anna_student_resnet18/scripts/evaluate_student.py

In Colab:

    !python students/anna_student_resnet18/scripts/evaluate_student.py
"""

from pathlib import Path
import sys

import torch


# Make local src/ importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from sixray_student.config import (
    CLASS_NAMES,
    LOAD_CHECKPOINT_PATH,
    EVAL_RESULTS_PATH,
    MAP_EVAL_CONFIDENCE_THRESHOLD,
    MAP_EVAL_NMS_IOU_THRESHOLD,
    MAP_EVAL_MAX_DETECTIONS,
)

from sixray_student.data import (
    build_datasets,
    build_dataloaders,
)

from sixray_student.model import (
    build_student_model,
    count_total_parameters,
    count_trainable_parameters,
)

from sixray_student.metrics import (
    evaluate_map,
    print_detection_metrics,
)

from sixray_student.train_utils import (
    evaluate_loss,
    load_checkpoint_for_training,
    save_json,
    to_jsonable,
)


def load_model_from_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a training checkpoint.

    Expected checkpoint format:
        {
            "model_state_dict": ...
        }

    This also accepts a raw state_dict as fallback.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint_for_training(
        path=checkpoint_path,
        device=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        loaded_epoch = checkpoint.get("epoch", None)
    else:
        model.load_state_dict(checkpoint)
        loaded_epoch = None

    return checkpoint, loaded_epoch


def evaluate_split(
    model,
    data_loader,
    device,
    split_name,
    use_amp,
):
    """
    Evaluate one split using both loss and detection mAP.
    """

    print("\n" + "=" * 80)
    print(f"Evaluating {split_name}")
    print("=" * 80)

    loss_metrics = evaluate_loss(
        model=model,
        data_loader=data_loader,
        device=device,
        use_amp=use_amp,
    )

    print(f"\n{split_name} loss:")
    print(loss_metrics)

    detection_metrics = evaluate_map(
        model=model,
        data_loader=data_loader,
        device=device,
        confidence_threshold=MAP_EVAL_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=MAP_EVAL_NMS_IOU_THRESHOLD,
        max_detections=MAP_EVAL_MAX_DETECTIONS,
        max_batches=None,
        use_amp=use_amp,
    )

    print_detection_metrics(
        detection_metrics,
        title=f"{split_name} detection metrics",
        class_names=CLASS_NAMES,
    )

    return {
        "loss": to_jsonable(loss_metrics),
        "detection": to_jsonable(detection_metrics),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Device:", device)
    print("Use AMP:", use_amp)

    print("\nBuilding datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets()

    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    print("\nBuilding model...")
    model = build_student_model()
    model = model.to(device)

    print("Total parameters:", count_total_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))

    print("\nLoading checkpoint:")
    print(LOAD_CHECKPOINT_PATH)

    checkpoint, loaded_epoch = load_model_from_checkpoint(
        model=model,
        checkpoint_path=LOAD_CHECKPOINT_PATH,
        device=device,
    )

    if loaded_epoch is not None:
        print("Loaded epoch:", loaded_epoch)
    else:
        print("Loaded checkpoint without epoch metadata.")

    model.eval()

    results = {
        "checkpoint_path": str(LOAD_CHECKPOINT_PATH),
        "loaded_epoch": loaded_epoch,
        "class_names": CLASS_NAMES,
        "confidence_threshold": MAP_EVAL_CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": MAP_EVAL_NMS_IOU_THRESHOLD,
        "max_detections": MAP_EVAL_MAX_DETECTIONS,
    }

    results["val"] = evaluate_split(
        model=model,
        data_loader=val_loader,
        device=device,
        split_name="val",
        use_amp=use_amp,
    )

    results["test"] = evaluate_split(
        model=model,
        data_loader=test_loader,
        device=device,
        split_name="test",
        use_amp=use_amp,
    )

    save_json(
        data=to_jsonable(results),
        path=EVAL_RESULTS_PATH,
    )

    print("\nSaved evaluation results:")
    print(EVAL_RESULTS_PATH)


if __name__ == "__main__":
    main()