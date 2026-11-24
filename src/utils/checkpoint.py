import json
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoConfig


def save_checkpoint(model, processor, save_dir, metrics=None, epoch=None, extra=None, optimizer=None, scheduler=None):
    """Save the Hugging Face model, processor, and optional training metadata."""
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

    # Save optimizer and scheduler state for resuming training.
    training_state = {}
    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()
    if training_state:
        torch.save(training_state, save_dir / "training_state.pt")

def load_checkpoint(save_dir, device=None, *, strict=False):
    """Load a checkpoint written by save_checkpoint."""
    save_dir = Path(save_dir)
    if strict:
        # Require complete weights and let Transformers handle versioned key conversions.
        processor = AutoImageProcessor.from_pretrained(save_dir, local_files_only=True)
        model, loading_info = AutoModelForObjectDetection.from_pretrained(
            save_dir, local_files_only=True, output_loading_info=True,
            ignore_mismatched_sizes=False,
        )
        problems = {key: loading_info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        ) if loading_info.get(key)}
        if problems:
            raise RuntimeError(f"Incompatible teacher checkpoint {save_dir}: {problems}")
        if device is not None:
            model = model.to(device)
        model.eval()
        meta_path = save_dir / "training_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return processor, model, meta

    processor = AutoImageProcessor.from_pretrained(save_dir)

    config = AutoConfig.from_pretrained(save_dir)
    model = AutoModelForObjectDetection.from_config(config)

    weights_path = save_dir / "model_full_weights.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Convert older checkpoint keys.
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('model.encoder.encoder.', 'model.encoder.aifi.')
            k = k.replace('.self_attn.out_proj.', '.self_attn.o_proj.')
            k = k.replace('.fc1.', '.mlp.fc1.')
            k = k.replace('.fc2.', '.mlp.fc2.')
            clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=True)

        print(f"Loaded full weights from {weights_path}")
    else:
        print(f"No full weights found in {weights_path}, loading Hugging Face weights")
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
    """Load optimizer and scheduler state to resume training."""
    save_dir = Path(save_dir)
    
    state_path = Path(save_dir) / "training_state.pt"
    if state_path.exists():
        return torch.load(state_path, map_location="cpu")

    # Older checkpoints store these states separately.
    state = {}
    if (save_dir / "optimizer_state_dict.pt").exists():
        state["optimizer"] = torch.load(save_dir / "optimizer_state_dict.pt", map_location="cpu")
    if (save_dir / "scheduler_state_dict.pt").exists():
        state["scheduler"] = torch.load(save_dir / "scheduler_state_dict.pt", map_location="cpu")
    return state
