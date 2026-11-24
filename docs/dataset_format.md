# Dataset and annotations

We use a supplied SIXray image subset and SIXray-D XML annotations. Images, annotation JSON files, and saved splits must be obtained separately. Use the files from the original experiment, including the same image ordering.

## File layout

```text
subset_clean/                 # some older notebooks use a different dataset root
  train.json
  train/images/...
  test.json
  test/images/...
```

Image filenames in JSON are relative to the corresponding `images/` directory. Training and validation use `train.json` and `train/images/`; test indices refer to `test.json` and `test/images/`.

## Annotation JSON

The teacher loader, `SixRayDataset`, requires `images`, `categories`, and `annotations`. The student loader, `SixRayStudentDataset`, expects flat annotations linked by `image_id`. Use the flat format when loading the same file with both implementations.

```json
{
  "images": [
    {"id": 12, "file_name": "pos/001.jpg", "width": 901, "height": 785},
    {"id": 13, "file_name": "neg/002.jpg", "width": 800, "height": 600}
  ],
  "categories": [
    {"id": 0, "name": "gun"},
    {"id": 1, "name": "knife"},
    {"id": 2, "name": "wrench"},
    {"id": 3, "name": "pliers"},
    {"id": 4, "name": "scissors"}
  ],
  "annotations": [
    {"image_id": 12, "category_id": 1, "bbox": [100, 120, 50, 80], "iscrowd": 0}
  ]
}
```

Image IDs must be unique. Each annotation must reference a listed image and class. Boxes are `[x_min, y_min, width, height]` in original-image pixels. A negative image has no flat annotation entries. Student class order is fixed as shown above; the teacher derives label maps from `categories`. Width and height are useful metadata; the loaders read actual dimensions from images.

The teacher also accepts a per-image layout, with one annotation group for each entry in `images`, in the same order:

```json
"annotations": [
  {"annotations": [{"bbox": [100, 120, 50, 80], "category_id": 1}]},
  {"annotations": []}
]
```

The student loader requires flat annotations. Keep the original annotation files and image ordering when reproducing a run.

## Saved split

Later experiments load `split_seed42_train10500_val300pos1200neg.json` with `train_indices`, `val_indices`, and `test_indices`. These are zero-based positions in the annotation file's `images` list, not image IDs. Train/validation positions refer to `train.json`; test positions refer to `test.json`.

For teacher training, we call `create_train_val_split` with seed 42, training total 10,500, and validation counts 300 positive/1,200 negative. Matching a saved split also requires the same dataset and image ordering.

## Preprocessing and deduplication

In [the deduplication notebook](../notebooks/01_dataset_deduplication.ipynb), we match XML records to image stems, combine annotations, and compare perceptual hashes. We explore Hamming distances up to 2, then use identical pHashes for final grouping. Identical hashes do not imply byte-identical files. We choose one image per group, giving priority to the physical test folder, then box count, class count, and filename. The final cell creates a stratified 10% validation split in memory but does not export the JSON files used for training.

The teacher uses its saved Hugging Face image processor, with training augmentations in `src/data/transforms.py`. The student resizes to 640 × 640, scales and clips boxes, drops degenerate boxes, and converts images to [0, 1] tensors. ImageNet normalization is inside the student model. Feature KD feeds resized [0, 1] images to the frozen teacher backbone.

See [reproduction instructions](reproduction.md) for required files and [experiment notes](experiment_notes.md) for open questions about the data and evaluation.
