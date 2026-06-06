"""Annotation JSON format detection and normalization."""

from __future__ import annotations


def is_per_image_format(annotations: list, images: list) -> bool:
    """True when annotations[i] holds boxes for images[i] (SixRay export)."""
    if not annotations:
        return True
    first = annotations[0]
    return "annotations" in first and "image_id" not in first


def index_annotations(data: dict) -> list[list[dict]]:
    """
    Normalize annotations to a per-image list aligned with data['images'].

    Supports:
      - SixRay per-image: annotations[i] == {"annotations": [bbox dicts]}
      - COCO flat:        annotations == [{image_id, bbox, category_id, ...}, ...]
    """
    images = data["images"]
    raw = data["annotations"]

    if is_per_image_format(raw, images):
        if len(raw) != len(images):
            raise ValueError(
                f"Per-image format requires len(annotations) == len(images), "
                f"got {len(raw)} vs {len(images)}."
            )
        return [entry.get("annotations", []) for entry in raw]

    by_image_id: dict[int, list[dict]] = {img["id"]: [] for img in images}
    for ann in raw:
        image_id = ann["image_id"]
        if image_id not in by_image_id:
            raise ValueError(f"Annotation references unknown image_id={image_id}.")
        by_image_id[image_id].append(ann)

    return [by_image_id[img["id"]] for img in images]
