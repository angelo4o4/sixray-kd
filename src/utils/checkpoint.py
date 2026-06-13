import json
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def save_checkpoint(model, processor, save_dir, metrics=None, epoch=None, extra=None, optimizer=None, scheduler=None):
    """Save HuggingFace model + processor and optional training metadata."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    meta = {}
    if epoch is not None:
        meta["epoch"] = epoch
    if metrics is not None:
        meta["metrics"] = metrics
    if extra:
        meta.update(extra)

    if meta:
        (save_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # saving also optimizer and scheduler state
    training_state = {}
    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()
    if training_state:
        torch.save(training_state, save_dir / "training_state.pt")

def load_checkpoint(save_dir, device=None):
    """Load a checkpoint written by save_checkpoint."""
    save_dir = Path(save_dir)
    processor = AutoImageProcessor.from_pretrained(save_dir)
    model = AutoModelForObjectDetection.from_pretrained(save_dir)
    if device is not None:
        model = model.to(device)

    meta_path = save_dir / "training_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return processor, model, meta

def load_training_state(save_dir):
    """Load optimizer and scheduler's state for the resume"""
    save_dir = Path(save_dir)
    
    state_path = Path(save_dir) / "training_state.pt"
    if state_path.exists():
        return torch.load(state_path, map_location="cpu")

    # Manual structure
    state = {}
    if (save_dir / "optimizer_state_dict.pt").exists():
        state["optimizer"] = torch.load(save_dir / "optimizer_state_dict.pt", map_location="cpu")
    if (save_dir / "scheduler_state_dict.pt").exists():
        state["scheduler"] = torch.load(save_dir / "scheduler_state_dict.pt", map_location="cpu")
    return state

def load_manual_checkpoint(save_dir, device=None):
    """Load chakpoint manually saved"""
    save_dir = Path(save_dir)

    processor = AutoImageProcessor.from_pretrained(save_dir / "processor")
    model = AutoModelForObjectDetection.from_pretrained(save_dir / "model_hf")
    if device is not None:
        model = model.to(device)

    meta_path = save_dir / "manual_metrics.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    epoch = meta.get("finished_epoch", meta.get("epoch", 0))

    map_val    = next((meta[k] for k in meta if "val_map" in k and "50" not in k and "75" not in k), 0.0)
    map_50_val = next((meta[k] for k in meta if "val_map_50" in k or "map_50" in k), 0.0)
    map_75_val = next((meta[k] for k in meta if "val_map_75" in k or "map_75" in k), 0.0)

    normalized = {
        "epoch": epoch,
        "metrics": {
            "map":    map_val,
            "map_50": map_50_val,
            "map_75": map_75_val,
        }
    }
    training_meta_path = save_dir / "training_meta.json"
    if not training_meta_path.exists():
        training_meta_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    return processor, model, normalized