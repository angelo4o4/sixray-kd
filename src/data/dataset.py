import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.labels import build_label_maps, categories_from_data
from src.data.schema import index_annotations


class SixRayDataset(Dataset):
    def __init__(self, image_dir, anno_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "images" not in data or "annotations" not in data:
            raise ValueError("Annotation JSON must contain 'images' and 'annotations' keys.")

        self.images = data["images"]
        self.categories = categories_from_data(data)
        self.id2label, self.label2id, self.num_labels = build_label_maps(self.categories)
        self.annotations_per_image = index_annotations(data)

        if len(self.annotations_per_image) != len(self.images):
            raise ValueError(
                f"Internal error: expected {len(self.images)} annotation groups, "
                f"got {len(self.annotations_per_image)}."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        anns = self.annotations_per_image[idx]

        image_path = os.path.join(self.image_dir, image_info["file_name"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        target = {"image_id": image_info["id"], "annotations": []}
        for ann in anns:
            bbox = ann["bbox"]
            if len(bbox) != 4:
                raise ValueError(f"bbox must have 4 values, got {bbox!r} for image_id={image_info['id']}.")
            target["annotations"].append({
                "bbox": bbox,
                "category_id": ann["category_id"],
                "area": bbox[2] * bbox[3],
                "iscrowd": ann.get("iscrowd", 0),
            })

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }
