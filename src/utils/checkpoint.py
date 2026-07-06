import json
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoConfig


def save_checkpoint(model, processor, save_dir, metrics=None, epoch=None, extra=None, optimizer=None, scheduler=None):
    """Save HuggingFace model + processor and optional training metadata."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    torch.save(model.state_dict(), save_dir / "model_full_weights.pt")

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

    config = AutoConfig.from_pretrained(save_dir)
    model = AutoModelForObjectDetection.from_config(config)

    weights_path = save_dir / "model_full_weights.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded full weights from {weights_path}")
    else:
        print(f"No full weights found in {weights_path}, loading from hf")
        model = AutoModelForObjectDetection.from_pretrained(save_dir)

    if device is not None:
        model = model.to(device)
    # set model to evaluation mode by default
    model.eval()

    # load metadata
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
