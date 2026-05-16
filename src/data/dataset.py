import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class SixRayDataset(Dataset):
    def __init__(self, image_dir, anno_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        with open(anno_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        anno_info = self.annotations[idx]

        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        anns = anno_info.get("annotations", [])
        target = {"image_id": image_info["id"], "annotations":[]}
        for ann in anns:
            target["annotations"].append({
                "bbox": ann["bbox"],  # format: [x_min, y_min, w, h]
                "category_id": ann["category_id"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0
            })

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0]
        }

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]

    # automatic padding by the processor
    return {
        "pixel_values": pixel_values,
        "labels": labels
        }