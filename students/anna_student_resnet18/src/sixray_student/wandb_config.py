"""
W&B configuration and logging helpers for Anna's ResNet18 student detector.

This file is intentionally separated from config.py.

Important:
- config.py contains only constants and paths.
- this file contains W&B-specific logic.
- importing this file should not immediately start a W&B run.
- the W&B API key is read from Colab userdata under the name "WANDB_API_KEY".
"""

import os

from sixray_student.config import (
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    GRAD_CLIP_NORM,
    BEST_CHECKPOINT_METRIC,
    MAP_EVAL_EVERY_N_EPOCHS,
    MAP_EVAL_MAX_BATCHES,
    MAP_EVAL_CONFIDENCE_THRESHOLD,
    MAP_EVAL_NMS_IOU_THRESHOLD,
    CHECKPOINT_DIR,
    MODEL_NAME,
    BBOX_ENCODING,
)


# =========================
# W&B switches
# =========================

USE_WANDB = True


# =========================
# W&B run identity
# =========================

WANDB_PROJECT = "sixray-kd"
WANDB_RUN_NAME = "student-resnet18-yolo2-local-offsets"

WANDB_API_KEY_SECRET_NAME = "WANDB_API_KEY"

# Store W&B run id in the checkpoint directory.
# This lets the same W&B run resume after Colab disconnects.
WANDB_RUN_ID_PATH = CHECKPOINT_DIR / "student_resnet18_run_id.txt"


def get_wandb_api_key():
    """
    Read the W&B API key.

    Priority:
    1. Colab userdata secret named WANDB_API_KEY.
    2. Environment variable WANDB_API_KEY.
    3. None.

    We import google.colab only inside this function because google.colab
    does not exist in normal local VS Code Python environments.
    """

    try:
        from google.colab import userdata

        key = userdata.get(WANDB_API_KEY_SECRET_NAME)

        if key:
            return key

    except ImportError:
        pass

    env_key = os.environ.get(WANDB_API_KEY_SECRET_NAME)

    if env_key:
        return env_key

    return None


def login_to_wandb():
    """
    Login to W&B.

    In Colab, this uses:
        userdata.get("WANDB_API_KEY")

    If the key is not found, wandb.login() falls back to the normal interactive login.
    """

    import wandb

    api_key = get_wandb_api_key()

    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()


def init_wandb():
    """
    Initialize and return a W&B run.

    Returns:
        W&B run object if USE_WANDB=True.
        None if USE_WANDB=False.
    """

    if not USE_WANDB:
        return None

    import wandb

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    login_to_wandb()

    if WANDB_RUN_ID_PATH.exists():
        wandb_run_id = WANDB_RUN_ID_PATH.read_text().strip()
    else:
        wandb_run_id = wandb.util.generate_id()
        WANDB_RUN_ID_PATH.write_text(wandb_run_id)

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        id=wandb_run_id,
        resume="allow",
        config={
            # Model
            "model": "ResNet18 one-stage detector student",
            "model_name": MODEL_NAME,
            "bbox_encoding": BBOX_ENCODING,

            # Architecture
            "image_size": IMAGE_SIZE,
            "grid_size": GRID_SIZE,
            "num_boxes": NUM_BOXES,
            "num_classes": NUM_CLASSES,

            # Training
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "grad_clip_norm": GRAD_CLIP_NORM,

            # Checkpointing
            "best_checkpoint_metric": BEST_CHECKPOINT_METRIC,

            # mAP evaluation
            "map_eval_every_n_epochs": MAP_EVAL_EVERY_N_EPOCHS,
            "map_eval_max_batches": MAP_EVAL_MAX_BATCHES,
            "map_eval_confidence_threshold": MAP_EVAL_CONFIDENCE_THRESHOLD,
            "map_eval_nms_iou_threshold": MAP_EVAL_NMS_IOU_THRESHOLD,
        },
    )

    return wandb_run


def log_to_wandb(wandb_run, values, step=None):
    """
    Log scalar values to W&B.

    Example:
        log_to_wandb(
            wandb_run,
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
            },
            step=epoch,
        )
    """

    if wandb_run is None:
        return

    wandb_run.log(values, step=step)


def finish_wandb(wandb_run):
    """
    Close the W&B run at the end of training.
    """

    if wandb_run is None:
        return

    wandb_run.finish()


def _to_float(value):
    """
    Convert common metric value types to plain Python float.

    Handles:
    - Python floats
    - NumPy scalars
    - PyTorch tensors
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu().item()

    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def _to_list(value):
    """
    Convert per-class metric containers to a Python list.

    torchmetrics may return tensors for per-class AP.
    This makes W&B logging robust.
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()

    if hasattr(value, "tolist"):
        value = value.tolist()

    return list(value)


def log_map_to_wandb(wandb_run, prefix, metrics, class_names, step=None):
    """
    Log detection mAP metrics to W&B.

    Expected metrics dictionary:
        {
            "map": ...,
            "map_50": ...,
            "map_75": ...,
            "map_per_class": [...]
        }

    Example prefix values:
        "val"
        "test"
    """

    if wandb_run is None:
        return

    log_dict = {
        f"{prefix}/map": _to_float(metrics["map"]),
        f"{prefix}/map_50": _to_float(metrics["map_50"]),
        f"{prefix}/map_75": _to_float(metrics["map_75"]),
    }

    class_maps = metrics.get("map_per_class", [])

    if class_maps is not None:
        class_maps = _to_list(class_maps)

        for class_id, class_name in enumerate(class_names):
            if class_id >= len(class_maps):
                continue

            value = _to_float(class_maps[class_id])

            # torchmetrics can use -1 when a class has no valid AP.
            # Do not log invalid per-class values.
            if value >= 0:
                log_dict[f"{prefix}/map_{class_name}"] = value

    wandb_run.log(log_dict, step=step)