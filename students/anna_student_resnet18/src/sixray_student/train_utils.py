"""
Training utilities for Anna's ResNet18 YOLO-style student detector.

This module contains reusable training helpers:
- one training epoch
- validation loss evaluation
- checkpoint loading/saving
- history helpers
- AMP scaler helper

It does not contain the full experiment script.
The actual script will live in:

    scripts/train_student.py
"""

import json
from contextlib import nullcontext
from pathlib import Path

import torch

from sixray_student.config import (
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
    CLASS_NAMES,
    GRAD_CLIP_NORM,
    LOG_EVERY_N_BATCHES,
    BEST_CHECKPOINT_METRIC,
    BBOX_ENCODING,
)

from sixray_student.losses import yolo_loss
from sixray_student.target_encoder import encode_targets_yolo


def make_grad_scaler(device, use_amp=True):
    """
    Create a GradScaler for mixed precision training.

    AMP is useful only on CUDA. On CPU, the scaler is disabled.
    """

    enabled = bool(use_amp and device.type == "cuda")

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, use_amp=True):
    """
    Return the correct autocast context.

    On CUDA:
        uses torch.amp.autocast

    On CPU:
        returns a no-op context
    """

    enabled = bool(use_amp and device.type == "cuda")

    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    return nullcontext()


def load_checkpoint_for_training(path, device):
    """
    Load a PyTorch checkpoint.

    weights_only=False is needed for checkpoints that contain optimizer state,
    scheduler state, history dictionaries, and metadata.
    """

    path = Path(path)

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_json(data, path):
    """
    Save Python data as formatted JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_jsonable(value):
    """
    Convert tensors and other numeric containers into JSON-friendly values.
    """

    if torch.is_tensor(value):
        value = value.detach().cpu()

        if value.numel() == 1:
            return float(value.item())

        return value.tolist()

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    return value


def _encoded_count(encoded_targets, new_key, old_key):
    """
    Read counters from encoded target dictionaries.

    The refactored code uses:
        num_assigned
        num_skipped

    The notebook used:
        assigned_objects
        skipped_collisions

    This helper accepts both.
    """

    if new_key in encoded_targets:
        return int(encoded_targets[new_key])

    if old_key in encoded_targets:
        return int(encoded_targets[old_key])

    return 0


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    scaler,
    use_amp=True,
    grad_clip_norm=GRAD_CLIP_NORM,
    log_every_n_batches=LOG_EVERY_N_BATCHES,
):
    """
    Train the model for one epoch.

    Returns averaged loss values and target assignment statistics.
    """

    model.train()

    total_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    total_cls = 0.0
    total_assigned = 0
    total_skipped = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)

        encoded = encode_targets_yolo(
            targets=targets,
            image_size=IMAGE_SIZE,
            grid_size=GRID_SIZE,
            num_boxes=NUM_BOXES,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device=device, use_amp=use_amp):
            predictions = model(images)
            loss, loss_dict = yolo_loss(predictions, encoded)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict["total_loss"]
        total_obj += loss_dict["objectness_loss"]
        total_box += loss_dict["bbox_loss"]
        total_cls += loss_dict["class_loss"]

        assigned = _encoded_count(
            encoded,
            new_key="num_assigned",
            old_key="assigned_objects",
        )

        skipped = _encoded_count(
            encoded,
            new_key="num_skipped",
            old_key="skipped_collisions",
        )

        total_assigned += assigned
        total_skipped += skipped

        if log_every_n_batches and (batch_idx + 1) % log_every_n_batches == 0:
            print(
                f"Epoch {epoch}, batch {batch_idx + 1}/{len(data_loader)} | "
                f"loss={loss_dict['total_loss']:.4f} | "
                f"obj={loss_dict['objectness_loss']:.4f} | "
                f"box={loss_dict['bbox_loss']:.4f} | "
                f"cls={loss_dict['class_loss']:.4f} | "
                f"assigned={assigned} | "
                f"skipped={skipped}"
            )

    n_batches = len(data_loader)

    return {
        "total_loss": total_loss / n_batches,
        "objectness_loss": total_obj / n_batches,
        "bbox_loss": total_box / n_batches,
        "class_loss": total_cls / n_batches,
        "assigned_objects": int(total_assigned),
        "skipped_collisions": int(total_skipped),
    }


@torch.no_grad()
def evaluate_loss(
    model,
    data_loader,
    device,
    use_amp=True,
):
    """
    Evaluate validation/test loss without updating model parameters.
    """

    model.eval()

    total_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    total_cls = 0.0
    total_assigned = 0
    total_skipped = 0

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)

        encoded = encode_targets_yolo(
            targets=targets,
            image_size=IMAGE_SIZE,
            grid_size=GRID_SIZE,
            num_boxes=NUM_BOXES,
            device=device,
        )

        with autocast_context(device=device, use_amp=use_amp):
            predictions = model(images)
            loss, loss_dict = yolo_loss(predictions, encoded)

        total_loss += loss_dict["total_loss"]
        total_obj += loss_dict["objectness_loss"]
        total_box += loss_dict["bbox_loss"]
        total_cls += loss_dict["class_loss"]

        total_assigned += _encoded_count(
            encoded,
            new_key="num_assigned",
            old_key="assigned_objects",
        )

        total_skipped += _encoded_count(
            encoded,
            new_key="num_skipped",
            old_key="skipped_collisions",
        )

    n_batches = len(data_loader)

    return {
        "total_loss": total_loss / n_batches,
        "objectness_loss": total_obj / n_batches,
        "bbox_loss": total_box / n_batches,
        "class_loss": total_cls / n_batches,
        "assigned_objects": int(total_assigned),
        "skipped_collisions": int(total_skipped),
    }


def get_best_detection_metric_from_history(
    history,
    metric_name=BEST_CHECKPOINT_METRIC,
):
    """
    Find the best validation detection metric from training history.
    """

    best_value = -float("inf")
    best_epoch = None

    for item in history:
        val_detection = item.get("val_detection")

        if not isinstance(val_detection, dict):
            continue

        value = val_detection.get(metric_name)

        if value is None:
            continue

        value = float(value)

        if value >= 0.0 and value > best_value:
            best_value = value
            best_epoch = item.get("epoch")

    return best_value, best_epoch


def get_best_loss_from_history(history):
    """
    Find the best validation loss from training history.
    """

    best_value = float("inf")
    best_epoch = None

    for item in history:
        val_loss = item.get("val", {}).get("total_loss")

        if val_loss is None:
            continue

        val_loss = float(val_loss)

        if val_loss < best_value:
            best_value = val_loss
            best_epoch = item.get("epoch")

    return best_value, best_epoch


def build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    train_metrics,
    val_metrics,
    val_detection_metrics,
    history,
    best_val_metric,
    best_metric_epoch,
    best_val_loss,
    best_loss_epoch,
    wandb_run=None,
):
    """
    Build a checkpoint dictionary with all information needed to resume training.
    """

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": to_jsonable(train_metrics),
        "val_metrics": to_jsonable(val_metrics),
        "val_detection_metrics": to_jsonable(val_detection_metrics),
        "history": to_jsonable(history),
        "image_size": IMAGE_SIZE,
        "grid_size": GRID_SIZE,
        "num_classes": NUM_CLASSES,
        "num_boxes": NUM_BOXES,
        "class_names": CLASS_NAMES,
        "bbox_encoding": BBOX_ENCODING,
        "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,
        "best_val_metric": float(best_val_metric),
        "best_metric_epoch": best_metric_epoch,
        "best_val_loss": float(best_val_loss),
        "best_loss_epoch": best_loss_epoch,
    }

    if wandb_run is not None:
        checkpoint["wandb_run_id"] = wandb_run.id

    return checkpoint


def save_checkpoint(checkpoint, path):
    """
    Save checkpoint to disk.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def restore_training_state(
    checkpoint,
    model,
    optimizer=None,
    scheduler=None,
):
    """
    Restore model, optimizer, and scheduler states from checkpoint.

    Returns:
        next epoch number
    """

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1

    return start_epoch


def initialize_training_state(
    model,
    optimizer,
    scheduler,
    device,
    resume_training,
    resume_checkpoint_path,
):
    """
    Initialize or resume training state.

    Returns:
        history,
        start_epoch,
        best_val_loss,
        best_loss_epoch,
        best_val_metric,
        best_metric_epoch
    """

    history = []
    start_epoch = 1
    best_val_loss = float("inf")
    best_loss_epoch = None
    best_val_metric = -float("inf")
    best_metric_epoch = None

    resume_checkpoint_path = Path(resume_checkpoint_path)

    if resume_training and resume_checkpoint_path.exists():
        checkpoint = load_checkpoint_for_training(
            path=resume_checkpoint_path,
            device=device,
        )

        start_epoch = restore_training_state(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        history = checkpoint.get("history", [])

        best_val_loss, best_loss_epoch = get_best_loss_from_history(history)

        best_val_metric, best_metric_epoch = get_best_detection_metric_from_history(
            history,
            metric_name=BEST_CHECKPOINT_METRIC,
        )

        if checkpoint.get("best_checkpoint_metric") == BEST_CHECKPOINT_METRIC:
            best_val_metric = float(
                checkpoint.get("best_val_metric", best_val_metric)
            )
            best_metric_epoch = checkpoint.get(
                "best_metric_epoch",
                best_metric_epoch,
            )

        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_loss_epoch = checkpoint.get("best_loss_epoch", best_loss_epoch)

        print("Resumed from:", resume_checkpoint_path)
        print("Previous epoch:", start_epoch - 1)
        print("Next epoch:", start_epoch)
        print(f"Best previous val loss: {best_val_loss} at epoch {best_loss_epoch}")
        print(
            f"Best previous val {BEST_CHECKPOINT_METRIC}: "
            f"{best_val_metric} at epoch {best_metric_epoch}"
        )

    else:
        print("No checkpoint resume. Training starts from epoch 1.")

    return {
        "history": history,
        "start_epoch": start_epoch,
        "best_val_loss": best_val_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_val_metric": best_val_metric,
        "best_metric_epoch": best_metric_epoch,
    }


def update_best_checkpoints(
    checkpoint,
    val_metrics,
    val_detection_metrics,
    best_val_loss,
    best_loss_epoch,
    best_val_metric,
    best_metric_epoch,
    epoch,
    best_loss_checkpoint_path,
    best_metric_checkpoint_path,
):
    """
    Save best-loss and best-metric checkpoints when they improve.

    Returns updated:
        best_val_loss,
        best_loss_epoch,
        best_val_metric,
        best_metric_epoch
    """

    current_val_loss = float(val_metrics["total_loss"])

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_loss_epoch = int(epoch)

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_loss_epoch"] = best_loss_epoch

        save_checkpoint(checkpoint, best_loss_checkpoint_path)

        print(f"saved best loss checkpoint | val_loss={best_val_loss:.6f}")

    if val_detection_metrics is not None:
        current_val_metric = val_detection_metrics.get(BEST_CHECKPOINT_METRIC)

        if current_val_metric is None:
            print(
                f"Metric {BEST_CHECKPOINT_METRIC} was not found, "
                "so metric-best checkpoint was not updated."
            )

        else:
            current_val_metric = float(current_val_metric)

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_metric_epoch = int(epoch)

                checkpoint["best_val_metric"] = best_val_metric
                checkpoint["best_metric_epoch"] = best_metric_epoch
                checkpoint["best_checkpoint_metric"] = BEST_CHECKPOINT_METRIC

                save_checkpoint(checkpoint, best_metric_checkpoint_path)

                print(
                    f"saved best metric checkpoint | "
                    f"{BEST_CHECKPOINT_METRIC}={best_val_metric:.6f}"
                )

    return {
        "best_val_loss": best_val_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_val_metric": best_val_metric,
        "best_metric_epoch": best_metric_epoch,
    }