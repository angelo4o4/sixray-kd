"""Data preparation, prediction, and plotting for notebook 05."""

import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from src.engine.evaluator import _to_metric_target
from sixray_student.config import CLASS_NAMES, IMAGE_SIZE
from sixray_student.metrics import decode_predictions


def prepare_validation_files(archive_path, data_dir, indices):
    """Extract the annotation file and selected images, preserving existing files."""
    data_dir = Path(data_dir)
    with ZipFile(archive_path) as archive:
        if not (data_dir / "train.json").exists():
            archive.extract("train.json", data_dir)
        images = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))["images"]
        for index in indices:
            member = "train/images/" + images[index]["file_name"]
            if not (data_dir / member).exists():
                archive.extract(member, data_dir)


@torch.inference_mode()
def predict_image(position, teacher_data, student_data, teachers, student, device,
                  confidence=0.001):
    """Return ground truth and all three predictions in original-pixel xyxy."""
    index = student_data.indices[position]
    info = teacher_data.images[index]
    with Image.open(Path(teacher_data.image_dir) / info["file_name"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    teacher_item = teacher_data[index]
    pixels = teacher_item["pixel_values"].unsqueeze(0).to(device)
    target = _to_metric_target([teacher_item["labels"]], torch.device("cpu"))[0]

    predictions = {}
    for name, (processor, model) in teachers.items():
        prediction = processor.post_process_object_detection(
            model(pixel_values=pixels), threshold=confidence,
            target_sizes=torch.tensor([[height, width]], device=device),
        )[0]
        keep = prediction["scores"].argsort(descending=True)[:100]
        prediction = {key: value[keep].cpu() for key, value in prediction.items()}
        prediction["boxes"][:, 0::2].clamp_(0, width)
        prediction["boxes"][:, 1::2].clamp_(0, height)
        predictions[name] = prediction

    student_image, _ = student_data[position]
    predictions["student"] = decode_predictions(
        student(student_image.unsqueeze(0).to(device)),
        confidence_threshold=confidence, nms_iou_threshold=0.5, max_detections=100,
    )[0]
    predictions["student"]["boxes"] *= torch.tensor(
        [width / IMAGE_SIZE, height / IMAGE_SIZE] * 2,
    )
    return {"image": image, "file_name": info["file_name"],
            "ground_truth": target, "predictions": predictions}


def plot_comparison(sample, confidence=0.25):
    """Show ground truth and each model on the original RGB image."""
    panels = {"ground_truth": sample["ground_truth"], **sample["predictions"]}
    fig, axes = plt.subplots(1, len(panels), figsize=(22, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for ax, (name, detection) in zip(axes, panels.items()):
        ax.imshow(sample["image"])
        scores = detection.get("scores", torch.ones(len(detection["boxes"])))
        for box, label, score in zip(detection["boxes"], detection["labels"], scores):
            if score < confidence:
                continue
            x1, y1, x2, y2 = box.tolist()
            label = int(label)
            color = colors(label)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor=color, linewidth=2))
            text = CLASS_NAMES[label]
            if "scores" in detection:
                text += f" {float(score):.2f}"
            ax.text(x1, y1, text, color="white", fontsize=8,
                    bbox={"facecolor": color, "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")
    fig.suptitle(sample["file_name"])
    return fig
