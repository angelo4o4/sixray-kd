# SIXray Project: Teacher Training Progress & Results

## Overview

This document summarizes the training methodology and performance results for the **RT-DETR (RT-DETR v2)** object detection model on the SIXray dataset. The training process was divided into two distinct phases to address class imbalance and improve the model's robustness to spatial variations.
The teacher model weights initialization are not random but they are from IMAGENET (or COCO, I have to check)

## Phase 1: Baseline Training

**Goal:** Establish a solid baseline model performance using standard configurations.

- **Epochs:** 1-30.
- **Best Epoch:** 13.
- **Methodology:** Standard training pipeline.
- **Performance:**
  - **Val mAP:** 0.6666
  - **mAP@50:** 0.7852



### Training Metrics (Phase 1)

Training Loss Baseline
Val_map_per_class_4 (scissors)

---



## Phase 2: Oversampling & Augmentation

**Goal:** Improve detection capabilities for underperforming classes (e.g., scissors, wrench) and introduce spatial robustness through more augmentations.

- **Epochs:** 14-30 (17 epochs).
- **Best Epoch:** 28.
- **Starting weights:** the run was launched as a resume of Phase 1, but the resume code restored only the optimizer, the LR schedule and the epoch counter, not the model weights. Phase 2 therefore started again from the pretrained `PekingU/rtdetr_v2_r50vd` weights (val mAP after its first epoch: 0.4433, against 0.6666 at the end of Phase 1).
- **Key Strategies:**
  - **Weighted Random Sampler:** Implemented oversampling to force the model to focus on rare/harder classes.
  - **Vertical Flip Augmentation:** Introduced vertical flipping (p=0.5) to increase spatial invariance.
  - **Bounding Box Clipping:** Developed a custom `_clip_bbox` logic to ensure augmented bounding boxes remained within image boundaries, preventing CUDA illegal memory access errors.



### Comparison Table

The two columns are two separately trained models, not the same model before and after Phase 2. They differ in sampling, augmentation, number of epochs and LR schedule, so the delta cannot be attributed to oversampling alone.


| Class         | Baseline (Phase 1) | Oversampling (Phase 2) | Delta (Improvement) |
| ------------- | ------------------ | ---------------------- | ------------------- |
| **Gun**       | 0.9145             | 0.9174                 | +0.0029             |
| **Knife**     | 0.7683             | 0.8168                 | +0.0485             |
| **Wrench**    | 0.5951             | 0.6573                 | +0.0622             |
| **Pliers**    | 0.5823             | 0.6096                 | +0.0273             |
| **Scissors**  | 0.4727             | 0.6582                 | +0.1855             |
| **Total mAP** | **0.6666**         | **0.7319**             | **+0.0653**         |




### Training Metrics (Phase 2)

Oversampling Training Progress
*(Chart showing mAP improvement after Phase 2 -> da fare)*

## Conclusion

The model trained with oversampling and the extended augmentation reaches a higher validation mAP than the baseline (**0.73 against 0.67**), with the largest gain on scissors, the rarest class. Because Phase 2 did not continue from the Phase 1 weights, this is a comparison between two training recipes; isolating the effect of oversampling would need a run that changes only that.

The final best model was saved at **Epoch 28**.