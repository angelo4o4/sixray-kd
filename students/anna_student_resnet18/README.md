# ResNet18 YOLO-style student detector

This folder contains a ResNet18-based student detector.

## Model

The student is a small single-stage object detector.

Main design:

- backbone: ResNet18 pretrained on ImageNet
- input size: 640 x 640 RGB
- prediction grid: 20 x 20
- boxes per grid cell: 2
- 5 classes: gun, knife, wrench, pliers, scissors
- bbox encoding: local cell offsets for box center x/y, global normalized width/height
- output tensor shape: `[B, 2 * (1 + 4 + 5), 20, 20] = [B, 20, 20, 20]`

Each predicted box slot contains:

- 1 objectness logit
- 4 bbox values
- 5 class logits

## Folder structure

```text
students/anna_student_resnet18/
├── notebooks/
│   └── yolo_resnet18_student_anna_fixed.ipynb
├── scripts/
│   ├── train_student.py
│   ├── evaluate_student.py
│   
└── src/
    └── sixray_student/
        ├── __init__.py
        ├── config.py
        ├── wandb_config.py
        ├── data.py
        ├── model.py
        ├── target_encoder.py
        ├── losses.py
        ├── metrics.py
        └── train_utils.py
