"""
Train Anna's ResNet18 YOLO-style student detector on SIXray-D.

Run from repository root:

    python students/anna_student_resnet18/scripts/train_student.py

In Colab:

    !python students/anna_student_resnet18/scripts/train_student.py
"""

from pathlib import Path
import random
import sys

import numpy as np
import torch


# Make local src/ importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from sixray_student.config import (
    SEED,
    CLASS_NAMES,
    CHECKPOINT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RESUME_TRAINING,
    RESUME_CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    BEST_LOSS_CHECKPOINT_PATH,
    LAST_CHECKPOINT_PATH,
    HISTORY_PATH,
    MAP_EVAL_EVERY_N_EPOCHS,
    MAP_EVAL_MAX_BATCHES,
    MAP_EVAL_CONFIDENCE_THRESHOLD,
    MAP_EVAL_NMS_IOU_THRESHOLD,
    MAP_EVAL_MAX_DETECTIONS,
    SCHEDULER_T_MAX,
    SCHEDULER_ETA_MIN,
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
    make_grad_scaler,
    train_one_epoch,
    evaluate_loss,
    initialize_training_state,
    build_checkpoint,
    save_checkpoint,
    save_json,
    update_best_checkpoints,
    to_jsonable,
)

from sixray_student.wandb_config import (
    init_wandb,
    finish_wandb,
    log_to_wandb,
    log_map_to_wandb,
)


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    """
    Build optimizer for student training.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


def build_scheduler(optimizer):
    """
    Build learning-rate scheduler.

    The notebook used cosine annealing logic.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_T_MAX,
        eta_min=SCHEDULER_ETA_MIN,
    )

    return scheduler


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = make_grad_scaler(device=device, use_amp=use_amp)

    print("\nInitializing W&B...")
    wandb_run = init_wandb()

    print("\nInitializing training state...")
    state = initialize_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        resume_training=RESUME_TRAINING,
        resume_checkpoint_path=RESUME_CHECKPOINT_PATH,
    )

    history = state["history"]
    start_epoch = state["start_epoch"]

    best_val_loss = state["best_val_loss"]
    best_loss_epoch = state["best_loss_epoch"]

    best_val_metric = state["best_val_metric"]
    best_metric_epoch = state["best_metric_epoch"]

    if start_epoch > NUM_EPOCHS:
        print(
            f"Start epoch is {start_epoch}, but NUM_EPOCHS is {NUM_EPOCHS}. "
            "Nothing to train."
        )
        finish_wandb(wandb_run)
        return

    print("\nStarting training...")
    print(f"Epochs: {start_epoch} -> {NUM_EPOCHS}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print("=" * 80)

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = evaluate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nTrain loss:")
        print(train_metrics)

        print("\nValidation loss:")
        print(val_metrics)

        print("Current LR:", current_lr)

        val_detection_metrics = None

        should_eval_map = (
            MAP_EVAL_EVERY_N_EPOCHS is not None
            and MAP_EVAL_EVERY_N_EPOCHS > 0
            and epoch % MAP_EVAL_EVERY_N_EPOCHS == 0
        )

        if should_eval_map:
            print("\nEvaluating validation mAP...")

            val_detection_metrics = evaluate_map(
                model=model,
                data_loader=val_loader,
                device=device,
                confidence_threshold=MAP_EVAL_CONFIDENCE_THRESHOLD,
                nms_iou_threshold=MAP_EVAL_NMS_IOU_THRESHOLD,
                max_detections=MAP_EVAL_MAX_DETECTIONS,
                max_batches=MAP_EVAL_MAX_BATCHES,
                use_amp=use_amp,
            )

            print_detection_metrics(
                val_detection_metrics,
                title=f"Validation detection metrics epoch {epoch}",
                class_names=CLASS_NAMES,
            )

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": to_jsonable(train_metrics),
            "val": to_jsonable(val_metrics),
            "val_detection": to_jsonable(val_detection_metrics),
        }

        history.append(epoch_record)

        log_values = {
            "epoch": epoch,
            "lr": current_lr,
            "train/total_loss": train_metrics["total_loss"],
            "train/objectness_loss": train_metrics["objectness_loss"],
            "train/bbox_loss": train_metrics["bbox_loss"],
            "train/class_loss": train_metrics["class_loss"],
            "train/assigned_objects": train_metrics["assigned_objects"],
            "train/skipped_collisions": train_metrics["skipped_collisions"],
            "val/total_loss": val_metrics["total_loss"],
            "val/objectness_loss": val_metrics["objectness_loss"],
            "val/bbox_loss": val_metrics["bbox_loss"],
            "val/class_loss": val_metrics["class_loss"],
            "val/assigned_objects": val_metrics["assigned_objects"],
            "val/skipped_collisions": val_metrics["skipped_collisions"],
        }

        log_to_wandb(
            wandb_run=wandb_run,
            values=log_values,
            step=epoch,
        )

        if val_detection_metrics is not None:
            log_map_to_wandb(
                wandb_run=wandb_run,
                prefix="val",
                metrics=val_detection_metrics,
                class_names=CLASS_NAMES,
                step=epoch,
            )

        checkpoint = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_detection_metrics=val_detection_metrics,
            history=history,
            best_val_metric=best_val_metric,
            best_metric_epoch=best_metric_epoch,
            best_val_loss=best_val_loss,
            best_loss_epoch=best_loss_epoch,
            wandb_run=wandb_run,
        )

        best_state = update_best_checkpoints(
            checkpoint=checkpoint,
            val_metrics=val_metrics,
            val_detection_metrics=val_detection_metrics,
            best_val_loss=best_val_loss,
            best_loss_epoch=best_loss_epoch,
            best_val_metric=best_val_metric,
            best_metric_epoch=best_metric_epoch,
            epoch=epoch,
            best_loss_checkpoint_path=BEST_LOSS_CHECKPOINT_PATH,
            best_metric_checkpoint_path=BEST_CHECKPOINT_PATH,
        )

        best_val_loss = best_state["best_val_loss"]
        best_loss_epoch = best_state["best_loss_epoch"]
        best_val_metric = best_state["best_val_metric"]
        best_metric_epoch = best_state["best_metric_epoch"]

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_loss_epoch"] = best_loss_epoch
        checkpoint["best_val_metric"] = best_val_metric
        checkpoint["best_metric_epoch"] = best_metric_epoch

        save_checkpoint(checkpoint, LAST_CHECKPOINT_PATH)
        save_json(to_jsonable(history), HISTORY_PATH)

        print("\nSaved last checkpoint:", LAST_CHECKPOINT_PATH)
        print("Saved history:", HISTORY_PATH)

    print("\nTraining finished.")
    print("Best validation loss:", best_val_loss, "at epoch", best_loss_epoch)
    print("Best validation metric:", best_val_metric, "at epoch", best_metric_epoch)

    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()