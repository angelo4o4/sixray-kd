# ResNet18 student detector

We first train a supervised student detector, then compare supervised learning with feature distillation from a frozen RT-DETRv2 teacher.

## Model

We use an ImageNet-pretrained ResNet18 backbone and a convolutional detection head. Inputs are 640 × 640 RGB tensors in [0, 1]; ImageNet normalization happens inside the model.

The output is `[B, 20, 20, 20]`: 20 channels on a 20 × 20 grid. Each cell has two slots, each with one objectness logit, four box values, and five class logits. Class IDs are 0 gun, 1 knife, 2 wrench, 3 pliers, 4 scissors. Centers use local cell offsets; width and height are normalized to the image size. We skip objects beyond the two-slot capacity of a cell.

Detection loss is objectness BCE + 10 × box Smooth L1 + class cross entropy. For feature KD, we use the union of ground-truth boxes projected onto the feature grid as foreground; other cells are background. We average squared distances between channel-normalized teacher/student features separately in each region, then combine foreground + 0.1 × background per image and average over the batch. Negative images contribute only 0.1 × background loss. We use the 512-to-2048 channel adapter only during training.

## Files

```text
notebooks/resnet18_student.ipynb      # original supervised experiment
scripts/train_student.py             # supervised training and resume
scripts/evaluate_student.py          # validation and test evaluation
scripts/run_matched_feature_kd.py     # matched experiments across seeds
src/sixray_student/
  config.py                         # paths and experiment settings
  data.py                           # flat annotation JSON and saved splits
  model.py                          # backbone and detection head
  target_encoder.py                 # grid targets
  losses.py                         # detection and feature-KD losses
  metrics.py                        # decoding, NMS, and mAP
  train_utils.py                    # epoch loops and checkpoints
  wandb_config.py                    # optional W&B logging
```

The [feature-KD notebook](../../notebooks/06_student_feature_kd.ipynb) explains the matched seed-42 comparison.

## Running the scripts

Run from the repository root after installing `requirements.txt` and obtaining the images, flat annotation JSON files, saved split, and checkpoints. Set paths in [students/anna_student_resnet18/src/sixray_student/config.py](src/sixray_student/config.py); defaults refer to Colab and Google Drive. W&B is enabled by default. Configure credentials or set `USE_WANDB = False` in [students/anna_student_resnet18/src/sixray_student/wandb_config.py](src/sixray_student/wandb_config.py).

```bash
python students/anna_student_resnet18/scripts/train_student.py
python students/anna_student_resnet18/scripts/evaluate_student.py
```

We use seed 42, batch size 2, AdamW with learning rate and weight decay 1e-4, and a 50-epoch cosine schedule for standalone training. We select the best detector by validation AP50. When resuming is enabled and a checkpoint is available, training continues to the configured total epoch count. Evaluation reports validation and test loss and detection metrics.

For matched KD, we select by mAP and use a different confidence threshold and coordinate system. See [reproduction instructions](../../docs/reproduction.md) for the multi-seed script and [experiment notes](../../docs/experiment_notes.md) for the evaluation settings.

## Results

In the original student experiment, we obtain best validation AP50 0.651264 at epoch 22 and full test mAP/AP50/AP75 of 0.2924/0.5537/0.2744. The quick validation check uses only 20 batches; the following cell evaluates the full validation and test splits.

In the matched seed-42 experiment, we obtain validation mAP 0.3813 for the supervised control and 0.4043 for feature KD. The [main README](../../README.md) includes the comparison with both teachers.
