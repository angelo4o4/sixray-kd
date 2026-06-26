# Anna ResNet18 YOLO-style student detector

This folder contains Anna's ResNet18-based student detector for the SIXray-D knowledge distillation project.

The code is kept inside a separate student workspace to avoid interfering with other team members' files.

## Model

The student is a small single-stage object detector.

Main design:

- backbone: ResNet18 pretrained on ImageNet
- input size: 640 x 640 RGB
- prediction grid: 20 x 20
- boxes per grid cell: 2
- classes: gun, knife, wrench, pliers, scissors
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
│   └── smoke_test_student.py
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