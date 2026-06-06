from transformers import AutoModelForObjectDetection, AutoImageProcessor


def load_teacher(
    model_name: str,
    id2label: dict,
    label2id: dict,
    device=None,
):
    """Load RT-DETR with a detection head sized for the dataset categories."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if device is not None:
        model = model.to(device)
    return processor, model
