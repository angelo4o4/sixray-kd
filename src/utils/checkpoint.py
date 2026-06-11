import json
from pathlib import Path

from transformers import AutoImageProcessor, AutoModelForObjectDetection


def save_checkpoint(model, processor, save_dir, metrics=None, epoch=None, extra=None):
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
