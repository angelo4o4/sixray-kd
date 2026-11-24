# Experiment notes

## Teacher checkpoint initialization

In notebook 04, we intended to continue from the baseline best checkpoint at epoch 13. The model cell calls `load_teacher`, which initializes from `PekingU/rtdetr_v2_r50vd`. The later call to `DetectionTrainer.fit(..., resume_from=RESUME_FROM)` restores the epoch number and optimizer/scheduler state, but does not load the checkpoint model weights.

We need to check the source version and checkpoint used for this experiment to confirm whether we started from the baseline weights.

## Baseline augmentation

The original notes for notebook 03 describe horizontal flips only. Its call to `build_train_transforms` supplies `flip_p` and `enabled`, but the current defaults also apply vertical flips at 0.5 and brightness/contrast jitter at 0.2. We need the source version used for that run to confirm its augmentation settings.

## Evaluation settings

| Experiment | Checkpoint selection | Confidence | Coordinates and split |
|---|---|---|---|
| Teacher training notebooks | Validation mAP | 0.1 | Original image coordinates; generated train/validation indices |
| Original student notebook and standalone scripts | Validation AP50 (`map_50`) | 0.05 | Resized 640 × 640 coordinates; the notebook also includes a 20-batch quick check |
| Notebook 05 | Previously selected checkpoints | 0.001 | Original image coordinates; saved validation split |
| Notebook 06 and matched runner | Validation mAP | 0.001 | Original image coordinates; saved validation split |

We use class-aware NMS at IoU 0.5 for the student and keep at most 100 detections. In notebook 05, we use teacher post-processing and keep the top 100 predictions. Matched evaluation disables AMP; standalone student evaluation uses AMP on CUDA. We report scores from these different evaluation settings separately.

For the early sanity check, we used a small validation sample from the supplied test folder. The later experiments use a separate validation split.

## Dataset and training environment

Teacher notebooks generate split indices from their dataset; later notebooks load a saved split. Matching the split requires the same seed, sample counts, images, and image ordering. The deduplication notebook stops before export, so it does not fully explain how the training archive `subset_clean.zip` was built.

Requirements do not pin all package versions used for training. The checkpoint loaders support older weight formats, but resuming the same training run also depends on the saved training state. The standalone student checkpoint does not save the AMP scaler or full RNG states; the matched runner saves the scaler state.

## Notebook display

Five notebooks contain widget-view outputs without saved widget states. Interactive progress bars or sliders may not work on GitHub, though static images and text are included.
