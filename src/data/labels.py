"""Label-map helpers for SixRay annotation JSON files."""

import json


def categories_from_data(data: dict) -> list[dict]:
    categories = data.get("categories")
    if not categories:
        raise ValueError(
            "Annotation JSON must include a 'categories' list with "
            "{'id': int, 'name': str} entries."
        )
    for cat in categories:
        if "id" not in cat or "name" not in cat:
            raise ValueError(f"Invalid category entry: {cat!r}")
    return categories


def build_label_maps(categories: list[dict]) -> tuple[dict, dict, int]:
    """Return (id2label, label2id, num_labels) for HuggingFace model config."""
    id2label = {str(cat["id"]): cat["name"] for cat in categories}
    label2id = {cat["name"]: cat["id"] for cat in categories}
    return id2label, label2id, len(categories)


def load_label_maps_from_file(anno_file: str) -> tuple[dict, dict, int]:
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = categories_from_data(data)
    return build_label_maps(categories)
