"""Train matched supervised and feature-KD students on SIXray-D.

Run from the same repository location as the original matched-run script.
The model, dataset, losses, and training loop remain in sixray_student.
"""

from datetime import datetime
from pathlib import Path
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "students/anna_student_resnet18/src")]

from src.engine.inference import prepare_validation_files
from src.utils.checkpoint import load_checkpoint
from sixray_student.data import SixRayStudentDataset, detection_collate_fn, load_split
from sixray_student.model import build_student_model
from sixray_student.metrics import evaluate_map
from sixray_student.train_utils import train_one_epoch, make_grad_scaler

ARTIFACTS = Path("C:/Users/89526/Documents/sixray-artifacts")
INITIAL_RUN = ARTIFACTS / "experiments/feature_kd_20261016T214646677055Z"
CONFIG_PATH = INITIAL_RUN / "kd/config.json"
INITIAL_WEIGHTS = INITIAL_RUN / "initial_student.pth"

EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
BACKGROUND_WEIGHT = 0.1
KD_WEIGHT = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_student(name, use_kd, config, initial_weights, teacher, train_data, val_data, output_dir):
    """Both runs start from identical student weights and data-shuffle seed."""
    set_seed(config["seed"])
    run_dir = output_dir / name
    run_dir.mkdir()

    student = build_student_model(pretrained_backbone=False).to(DEVICE)
    student.load_state_dict(initial_weights)

    adapter = None
    parameters = list(student.parameters())
    if use_kd:
        with torch.random.fork_rng():
            torch.manual_seed(config["adapter_seed"])
            adapter = torch.nn.Conv2d(512, 2048, kernel_size=1, bias=False).to(DEVICE)
        parameters += list(adapter.parameters())

    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=2, collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=detection_collate_fn,
    )

    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE,
    )
    scaler = make_grad_scaler(DEVICE)

    best_map = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        training = train_one_epoch(
            student, train_loader, optimizer, DEVICE, epoch, scaler,
            use_amp=True, log_every_n_batches=1000,
            teacher=teacher if use_kd else None,
            adapter=adapter, kd_weight=KD_WEIGHT,
            background_weight=BACKGROUND_WEIGHT, scheduler=scheduler,
        )
        validation = evaluate_map(
            student, val_loader, DEVICE,
            confidence_threshold=0.001, nms_iou_threshold=0.5,
            max_detections=100, use_amp=False, original_coordinates=True,
        )
        history.append({"epoch": epoch, "train": training, "validation": validation})

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
            "validation": validation,
        }
        if validation["map"] > best_map:
            best_map = validation["map"]
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "student_best_map.pth")
        torch.save(checkpoint, run_dir / "student_last.pth")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"{name:7} | epoch {epoch:02}/{EPOCHS} | val mAP {validation['map']:.4f} | best {best_map:.4f}", flush=True)

    return {"run": name, "best_epoch": best_epoch, "best_map": best_map}


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("This experiment was designed for CUDA training.")

    config = json.loads(CONFIG_PATH.read_text())
    initial_weights = torch.load(INITIAL_WEIGHTS, map_location="cpu", weights_only=True)
    output_dir = ARTIFACTS / "experiments" / f"matched_simple_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "settings.json").write_text(json.dumps({
        "seed": config["seed"], "adapter_seed": config["adapter_seed"],
        "split_path": config["split_path"], "teacher_path": config["teacher_path"],
        "initial_weights": str(INITIAL_WEIGHTS), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "background_weight": BACKGROUND_WEIGHT, "kd_weight": KD_WEIGHT,
        "scheduler": "CosineAnnealingLR(T_max=50); step timing requires verification",
    }, indent=2))

    data_dir = ARTIFACTS / "DatasetAPAI"
    train_indices, val_indices, _ = load_split(config["split_path"])
    prepare_validation_files(data_dir / "subset_clean.zip", data_dir / "subset_clean", train_indices + val_indices)
    images_dir = data_dir / "subset_clean/train/images"
    annotations = data_dir / "subset_clean/train.json"
    train_data = SixRayStudentDataset(images_dir, annotations, train_indices)
    val_data = SixRayStudentDataset(images_dir, annotations, val_indices)

    _, teacher, _ = load_checkpoint(config["teacher_path"], device=DEVICE, strict=True)
    teacher.requires_grad_(False).eval()

    results = []
    for name, use_kd in (("control", False), ("kd", True)):
        results.append(train_student(
            name, use_kd, config, initial_weights, teacher,
            train_data, val_data, output_dir,
        ))
    (output_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
