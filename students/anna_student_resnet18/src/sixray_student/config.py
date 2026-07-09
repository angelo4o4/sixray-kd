"""
Configuration for Anna's ResNet18 YOLO-style student detector.

This file should contain only constants and paths.

It should NOT:
- create datasets
- initialize models
- start W&B
- create optimizers
- run training
- load checkpoints

Those things belong in scripts or utility modules.
"""

from pathlib import Path


# =========================
# Reproducibility
# =========================

SEED = 42


# =========================
# Dataset paths
# =========================
# These paths match the Colab notebook setup.
# They are intentionally kept as Path objects.

DATA_ROOT = Path("/content/data")

TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
TRAIN_JSON = DATA_ROOT / "train.json"

TEST_IMG_DIR = DATA_ROOT / "test" / "images"
TEST_JSON = DATA_ROOT / "test.json"

SPLIT_PATH = Path(
    "/content/drive/MyDrive/DatasetAPAI/SIXray_Project/splits/"
    "split_seed42_train10500_val300pos1200neg.json"
)


# =========================
# Classes
# =========================

CLASS_NAMES = [
    "gun",
    "knife",
    "wrench",
    "pliers",
    "scissors",
]

NUM_CLASSES = len(CLASS_NAMES)

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# =========================
# Model
# =========================

MODEL_NAME = "student_resnet18_yolo2_local_offsets"

IMAGE_SIZE = 640
GRID_SIZE = 20

NUM_BOXES = 2
HEAD_CHANNELS = 256
PRETRAINED_BACKBONE = True

# The model predicts:
# - tx, ty as local offsets inside the assigned grid cell
# - w, h as global normalized image width/height
BBOX_ENCODING = "local_cell_offsets_xy_global_wh"


# =========================
# Dataloader
# =========================

BATCH_SIZE = 2
NUM_WORKERS = 2
PIN_MEMORY = True


# =========================
# Training
# =========================

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Total epoch number to reach.
# Example: if the last checkpoint is epoch 30 and NUM_EPOCHS = 50,
# training resumes from epoch 31 and stops after epoch 50.
NUM_EPOCHS = 50

GRAD_CLIP_NORM = 1.0

RUN_TRAINING = True
RESUME_TRAINING = True

LOG_EVERY_N_BATCHES = 100


# =========================
# Optimizer / scheduler
# =========================

OPTIMIZER_NAME = "AdamW"

SCHEDULER_NAME = "CosineAnnealingLR"
SCHEDULER_T_MAX = NUM_EPOCHS
SCHEDULER_ETA_MIN = 1e-6


# =========================
# Loss
# =========================

BOX_LOSS_WEIGHT = 10.0

# Objectness is very imbalanced because most grid cells are empty.
# The positive class weight is capped to avoid unstable gradients.
MAX_OBJECTNESS_POS_WEIGHT = 50.0


# =========================
# Checkpoints
# =========================

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/DatasetAPAI/SIXray_Project/student_checkpoints_anna"
)

# Main checkpoint is selected by detection quality mAP@50.
BEST_CHECKPOINT_METRIC = "map_50"

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"{MODEL_NAME}_best_{BEST_CHECKPOINT_METRIC}.pth"
)


BEST_LOSS_CHECKPOINT_PATH = CHECKPOINT_DIR / "student_resnet18_best_loss.pth"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "student_resnet18_last.pth"

HISTORY_PATH = CHECKPOINT_DIR / "student_resnet18_history.json"
EVAL_RESULTS_PATH = CHECKPOINT_DIR / "student_resnet18_eval_metrics.json"

RESUME_CHECKPOINT_PATH = LAST_CHECKPOINT_PATH


# =========================
# mAP evaluation during training
# =========================

MAP_EVAL_EVERY_N_EPOCHS = 1

# None means full validation set.
MAP_EVAL_MAX_BATCHES = None

MAP_EVAL_CONFIDENCE_THRESHOLD = 0.05
MAP_EVAL_NMS_IOU_THRESHOLD = 0.50
MAP_EVAL_MAX_DETECTIONS = 100


# =========================
# Final evaluation
# =========================

LOAD_CHECKPOINT_PATH = BEST_CHECKPOINT_PATH