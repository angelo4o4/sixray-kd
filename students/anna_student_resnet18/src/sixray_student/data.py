"""
Dataset and DataLoader utilities for Anna's ResNet18 student detector.

This module contains only data-related code:
- reading the train/val/test split
- loading SIXray-D images and annotations
- resizing images to the student input size
- converting COCO-style bbox annotations to xyxy tensors
- building PyTorch DataLoaders

It intentionally does not contain model, loss, metric, or visualization code.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from sixray_student.config import (
    IMAGE_SIZE,
    TRAIN_IMG_DIR,
    TRAIN_JSON,
    TEST_IMG_DIR,
    TEST_JSON,
    SPLIT_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


class SixRayStudentDataset(Dataset):
    """
    Dataset for the ResNet18 YOLO-style student detector.

    Expected annotation format:
        {
            "images": [
                {
                    "id": int,
                    "file_name": str,
                    ...
                }
            ],
            "annotations": [
                {
                    "image_id": int,
                    "bbox": [x, y, width, height],
                    "category_id": int,
                    ...
                }
            ]
        }

    The split file stores indices into the "images" list, not image ids.
    This is why __getitem__ first maps dataset_index -> real_index -> image_info.
    """

    def __init__(self, images_dir, annotation_file, indices, image_size=IMAGE_SIZE):
        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)
        self.indices = list(indices)
        self.image_size = int(image_size)

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

        # Group annotations by image id so __getitem__ does not scan all annotations.
        self.annotations_by_image_id = defaultdict(list)
        for annotation in self.annotations:
            self.annotations_by_image_id[annotation["image_id"]].append(annotation)

        print("Dataset:", self.annotation_file)
        print("Images folder:", self.images_dir)
        print("Images used here:", len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, dataset_index):
        real_index = self.indices[dataset_index]

        image_info = self.images[real_index]
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = self.images_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        old_width, old_height = image.size

        image = image.resize((self.image_size, self.image_size))
        image_tensor = TF.to_tensor(image)

        scale_x = self.image_size / old_width
        scale_y = self.image_size / old_height

        boxes = []
        labels = []

        for annotation in self.annotations_by_image_id.get(image_id, []):
            x, y, width, height = annotation["bbox"]

            x1 = float(x) * scale_x
            y1 = float(y) * scale_y
            x2 = float(x + width) * scale_x
            y2 = float(y + height) * scale_y

            # Keep boxes inside the resized image.
            x1 = max(0.0, min(x1, self.image_size))
            y1 = max(0.0, min(y1, self.image_size))
            x2 = max(0.0, min(x2, self.image_size))
            y2 = max(0.0, min(y2, self.image_size))

            label = int(annotation["category_id"])

            # Skip degenerate boxes.
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        }

        return image_tensor, target


def detection_collate_fn(batch):
    """
    Collate function for object detection.

    Images can be stacked into one tensor:
        images: [B, 3, H, W]

    Targets must stay as a list because each image has a different number of boxes:
        targets: list[dict]
    """

    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


def load_split(split_path=SPLIT_PATH):
    """
    Load train/val/test image indices from the split JSON file.
    """

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    test_indices = split_info["test_indices"]

    print("Loaded split:", split_path)
    print("Train:", len(train_indices))
    print("Val:", len(val_indices))
    print("Test:", len(test_indices))

    return train_indices, val_indices, test_indices


def build_datasets(
    train_img_dir=TRAIN_IMG_DIR,
    train_json=TRAIN_JSON,
    test_img_dir=TEST_IMG_DIR,
    test_json=TEST_JSON,
    split_path=SPLIT_PATH,
    image_size=IMAGE_SIZE,
):
    """
    Build train, validation, and test datasets.

    Important:
    - train and val both use TRAIN_JSON / TRAIN_IMG_DIR
    - test uses TEST_JSON / TEST_IMG_DIR

    This matches the notebook logic.
    """

    train_indices, val_indices, test_indices = load_split(split_path)

    train_dataset = SixRayStudentDataset(
        images_dir=train_img_dir,
        annotation_file=train_json,
        indices=train_indices,
        image_size=image_size,
    )

    val_dataset = SixRayStudentDataset(
        images_dir=train_img_dir,
        annotation_file=train_json,
        indices=val_indices,
        image_size=image_size,
    )

    test_dataset = SixRayStudentDataset(
        images_dir=test_img_dir,
        annotation_file=test_json,
        indices=test_indices,
        image_size=image_size,
    )

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    """
    Build PyTorch DataLoaders for train, validation, and test.

    pin_memory is enabled only if CUDA is available.
    This avoids unnecessary pinned-memory behavior on local CPU runs.
    """

    use_pin_memory = bool(pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader


def find_first_positive_indices(dataset, n=5):
    """
    Find the first dataset indices with at least one ground-truth box.

    Useful for quick sanity checks and visualization.
    """

    positive_indices = []

    for dataset_index in range(len(dataset)):
        _, target = dataset[dataset_index]

        if len(target["boxes"]) > 0:
            positive_indices.append(dataset_index)

        if len(positive_indices) >= n:
            break

    return positive_indices


def find_first_negative_indices(dataset, n=5):
    """
    Find the first dataset indices with no ground-truth boxes.

    Useful for checking negative SIXray images.
    """

    negative_indices = []

    for dataset_index in range(len(dataset)):
        _, target = dataset[dataset_index]

        if len(target["boxes"]) == 0:
            negative_indices.append(dataset_index)

        if len(negative_indices) >= n:
            break

    return negative_indices