# SixRay annotation JSON format

`SixRayDataset` expects a single JSON file per split (e.g. `train.json`, `test.json`) with three top-level keys.

## Required structure

```json
{
  "images": [
    {"id": 12, "file_name": "pos/001.jpg", "width": 901, "height": 785}
  ],
  "categories": [
    {"id": 0, "name": "gun"},
    {"id": 1, "name": "knife"}
  ],
  "annotations": []
}
```

### `images`

One entry per image. Required fields:

| Field        | Type | Description                          |
|--------------|------|--------------------------------------|
| `id`         | int  | Unique image id                      |
| `file_name`  | str  | Path relative to the images directory |

Optional but recommended: `width`, `height`.

### `categories`

Class definitions used to build the detection head (`num_labels`, `id2label`, `label2id`).
Every `category_id` in a bounding box must appear here.

SixRay typically uses five classes: gun, knife, wrench, pliers, scissors.

### `annotations` (two supported layouts)

#### 1. Per-image format (SixRay export)

`annotations` has the **same length** as `images`. Entry `i` holds all boxes for `images[i]`:

```json
"annotations": [
  {
    "annotations": [
      {"bbox": [100, 120, 50, 80], "category_id": 1}
    ]
  },
  {
    "annotations": []
  }
]
```

Negative images use an empty inner list.

#### 2. COCO flat format

Standard COCO detection layout — one list entry per box:

```json
"annotations": [
  {"image_id": 12, "category_id": 1, "bbox": [100, 120, 50, 80], "iscrowd": 0}
]
```

`bbox` is always `[x_min, y_min, width, height]` in **absolute pixels**.

## File layout on disk

```
subset/
  train/
    images/
      pos/001.jpg
      ...
  train.json
  test/
    images/
      ...
  test.json
```

Pass `image_dir=.../train/images` and `anno_file=.../train.json` to `SixRayDataset`.
