from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.engine.evaluator import evaluate_detection
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import NullLogger


class DetectionTrainer:
    def __init__(
        self,
        model,
        processor,
        device,
        checkpoint_dir,
        run_name="rtdetr",
        lr=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        eval_score_threshold=0.1,
        logger=None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eval_score_threshold = eval_score_threshold
        self.logger = logger or NullLogger()

    def fit(self, train_loader, val_loader, epochs):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"Starting training for {epochs} epochs")
        print(f"Total steps: {num_training_steps} | Warmup steps: {num_warmup_steps}")

        history = {"train_loss": [], "val_map": [], "val_map_50": [], "val_map_75": []}
        best_map = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch, epochs)
            val_metrics = evaluate_detection(
                self.model,
                self.processor,
                val_loader,
                self.device,
                score_threshold=self.eval_score_threshold,
            )

            history["train_loss"].append(train_loss)
            history["val_map"].append(val_metrics["map"])
            history["val_map_50"].append(val_metrics["map_50"])
            history["val_map_75"].append(val_metrics["map_75"])

            print(f"End of epoch {epoch + 1} - Average Loss: {train_loss:.4f}")
            print(
                f"  Val mAP: {val_metrics['map']:.4f} | "
                f"mAP@50: {val_metrics['map_50']:.4f} | "
                f"mAP@75: {val_metrics['map_75']:.4f}"
            )

            self.logger.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/map": val_metrics["map"],
                    "val/map_50": val_metrics["map_50"],
                    "val/map_75": val_metrics["map_75"],
                },
                step=epoch + 1,
            )

            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                best_epoch = epoch + 1
                save_dir = f"{self.checkpoint_dir}/{self.run_name}_best"
                save_checkpoint(
                    self.model,
                    self.processor,
                    save_dir,
                    metrics=val_metrics,
                    epoch=epoch + 1,
                )
                print(
                    f"New best model at epoch {epoch + 1} (mAP: {best_map:.4f}). "
                    f"Saved to {save_dir}"
                )

        self.logger.finish()
        print("Training finished!")
        print(f"Best val mAP: {best_map:.4f} at epoch {best_epoch}")
        history["best_map"] = best_map
        history["best_epoch"] = best_epoch
        return history

    def _train_epoch(self, train_loader, optimizer, scheduler, epoch, epochs):
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Training)")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{current_lr:.6f}")

        return epoch_loss / len(train_loader)
