"""Train matched supervised and feature-KD students on SIXray-D.

Run from the repository root; see docs/reproduction.md for required files.
"""

from datetime import datetime
from pathlib import Path
import json
import random
import sys
import shutil
import time
import traceback
from importlib.metadata import version

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

ARTIFACTS = ROOT / "artifacts"
INITIAL_RUN = ARTIFACTS / "experiments/feature_kd_20261016T214646677055Z"
INITIAL_WEIGHTS = INITIAL_RUN / "initial_student.pth"
EXISTING_PAIR = ARTIFACTS / "experiments/matched_20261016T221317947683Z"

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
    """We start both runs from the same student weights and shuffle seed."""
    set_seed(config["seed"])
    run_dir = output_dir / name
    run_dir.mkdir()
    config = {**config, "run": name, "kd_lambda": KD_WEIGHT if use_kd else 0.0}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

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
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE,
    )
    scaler = make_grad_scaler(DEVICE)

    best_map = -1.0
    best_epoch = 0
    history = []

    print(f"Starting seed {config['seed']} {name}: CUDA AMP, {EPOCHS} epochs", flush=True)
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
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
        history.append({"epoch": epoch, "train": training, "validation": validation,
                        "seconds": time.perf_counter() - started})

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
            "validation": validation,
            "config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        if validation["map"] > best_map:
            best_map = validation["map"]
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "student_best_map.pth")
        torch.save(checkpoint, run_dir / "student_last.pth")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"{name:7} | epoch {epoch:02}/{EPOCHS} | val mAP {validation['map']:.4f} | best {best_map:.4f}", flush=True)

    result = {"run": name, "best_epoch": best_epoch, "best_map": best_map}
    (run_dir / "complete.json").write_text(json.dumps(result, indent=2))
    return result


def prepare(output_dir):
    """Create all initial states and check pairing once, before any training."""
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((EXISTING_PAIR / "kd/config.json").read_text())
    previous = torch.load(INITIAL_WEIGHTS, map_location="cpu", weights_only=True)
    set_seed(42)
    reference = build_student_model(pretrained_backbone=True)
    assert all(torch.equal(v, previous[k]) for k, v in reference.state_dict().items())
    del reference
    initial_states = [previous]
    checks = []
    for seed in (43, 44, 45, 46):
        pair_dir = output_dir / f"seed_{seed}"
        pair_dir.mkdir()
        set_seed(seed)
        initial_student = build_student_model(pretrained_backbone=True)
        initial = initial_student.state_dict()
        assert all(any(not torch.equal(initial[k], state[k]) for k in initial) for state in initial_states)
        torch.save(initial, pair_dir / "initial_student.pth")
        control = build_student_model(pretrained_backbone=False)
        kd = build_student_model(pretrained_backbone=False)
        control.load_state_dict(initial)
        kd.load_state_dict(initial)
        assert all(torch.equal(control.state_dict()[k], kd.state_dict()[k]) for k in initial)
        generator = torch.Generator().manual_seed(seed)
        order_before = torch.randperm(10500, generator=generator)
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1)
            adapter = torch.nn.Conv2d(512, 2048, 1, bias=False).to(DEVICE)
        assert torch.equal(rng_before, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
        generator.manual_seed(seed)
        assert torch.equal(order_before, torch.randperm(10500, generator=generator))
        assert all(torch.equal(initial[k], kd.state_dict()[k]) for k in initial)
        pair_config = {**config, "seed": seed, "adapter_seed": seed + 1,
                       "initial_state_path": str(pair_dir / "initial_student.pth"),
                       "initial_state_sha256": None, "source_sha256": None,
                       "versions": {p: version(p) for p in
                                    ("torch", "torchvision", "transformers", "torchmetrics", "pycocotools")},
                       "python": sys.version, "gpu": torch.cuda.get_device_name(),
                       "process_id": None, "existing_seed_42": str(EXISTING_PAIR)}
        (pair_dir / "config.json").write_text(json.dumps(pair_config, indent=2))
        checks.append({"seed": seed, "identical_pair": True, "different_initialization": True,
                       "adapter_preserves_rng_and_student": True, "unique_output": str(pair_dir)})
        initial_states.append({k: v.clone() for k, v in initial.items()})
        del initial_student, control, kd, adapter
    # Save the script and student modules alongside the results.
    shutil.copy2(__file__, output_dir / "run_matched_feature_kd.py")
    shutil.copytree(ROOT / "students/anna_student_resnet18/src/sixray_student",
                    output_dir / "source/sixray_student", ignore=shutil.ignore_patterns("__pycache__"))
    (output_dir / "checks_passed.json").write_text(json.dumps(checks, indent=2))
    print("Preparation passed:", output_dir, flush=True)


def write_report(output_dir):
    rows = []
    for seed, pair_dir in [(42, EXISTING_PAIR)] + [(s, output_dir / f"seed_{s}") for s in (43, 44, 45, 46)]:
        row = {"seed": seed}
        for name in ("control", "kd"):
            history = json.loads((pair_dir / name / "history.json").read_text())
            best = max(history, key=lambda entry: entry["validation"]["map"])
            row[name] = {"epoch": best["epoch"], "checkpoint": str(pair_dir / name / "student_best_map.pth"),
                         **best["validation"]}
        row["paired_map_improvement"] = row["kd"]["map"] - row["control"]["map"]
        rows.append(row)
    summaries = {}
    for metric in ("map", "map_50", "map_75", "map_per_class"):
        control = np.array([r["control"][metric] for r in rows])
        kd = np.array([r["kd"][metric] for r in rows])
        summaries[metric] = {label: {"mean": values.mean(axis=0).tolist(),
                                           "sample_std": values.std(axis=0, ddof=1).tolist()}
                             for label, values in (("control", control), ("kd", kd), ("paired_improvement", kd-control))}
    (output_dir / "five_seed_report.json").write_text(json.dumps({"runs": rows, "summary": summaries,
        "class_order": ["gun", "knife", "wrench", "pliers", "scissors"],
        "settings": json.loads((output_dir / "seed_43/config.json").read_text())}, indent=2))
    lines = ["# Five-seed validation comparison", "", "| Seed | Control mAP | KD mAP | Difference | Best epochs control/KD |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['control']['map']:.6f} | {r['kd']['map']:.6f} | {r['paired_map_improvement']:+.6f} | {r['control']['epoch']}/{r['kd']['epoch']} |")
    lines += ["", "Mean and sample standard deviation (ddof=1):", "", "```json", json.dumps(summaries, indent=2), "```",
              "", "Full per-seed AP50, AP75, per-class AP, checkpoints and settings: five_seed_report.json."]
    (output_dir / "five_seed_report.md").write_text("\n".join(lines))


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("This experiment was designed for CUDA training.")

    output_dir = Path(sys.argv[2])
    if sys.argv[1] == "--prepare-only":
        prepare(output_dir)
        return
    if sys.argv[1] != "--run" or not (output_dir / "checks_passed.json").exists():
        raise ValueError("Use --prepare-only DIRECTORY, then --run DIRECTORY")
    # Create the marker once to prevent duplicate runs in this directory.
    (output_dir / "started.json").open("x").write(json.dumps({"started": datetime.now().isoformat()}))
    config = json.loads((output_dir / "seed_43/config.json").read_text())

    data_dir = ARTIFACTS / "DatasetAPAI"
    train_indices, val_indices, _ = load_split(config["split_path"])
    prepare_validation_files(data_dir / "subset_clean.zip", data_dir / "subset_clean", train_indices + val_indices)
    images_dir = data_dir / "subset_clean/train/images"
    annotations = data_dir / "subset_clean/train.json"
    train_data = SixRayStudentDataset(images_dir, annotations, train_indices)
    val_data = SixRayStudentDataset(images_dir, annotations, val_indices)

    _, teacher, _ = load_checkpoint(config["teacher_path"], device=DEVICE, strict=True)
    teacher.requires_grad_(False).eval()

    for seed in (43, 44, 45, 46):
        pair_dir = output_dir / f"seed_{seed}"
        config = json.loads((pair_dir / "config.json").read_text())
        initial_weights = torch.load(pair_dir / "initial_student.pth", map_location="cpu", weights_only=True)
        for name, use_kd in (("control", False), ("kd", True)):
            (output_dir / "status.json").write_text(json.dumps({"seed": seed, "run": name, "state": "running"}))
            train_student(name, use_kd, config, initial_weights, teacher, train_data, val_data, pair_dir)
    write_report(output_dir)
    (output_dir / "status.json").write_text(json.dumps({"state": "complete"}))
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if len(sys.argv) > 2 and Path(sys.argv[2]).is_dir():
            (Path(sys.argv[2]) / "failure.txt").write_text(traceback.format_exc())
        raise
