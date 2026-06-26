"""
This module contains:
- ImageNet normalization layer
- ResNet18 feature extractor
- YOLO-style prediction head
- parameter counting helpers

It does not contain:
- dataset loading
- target encoding
- losses
- metrics
- training loop
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from sixray_student.config import (
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_CLASSES,
    NUM_BOXES,
    HEAD_CHANNELS,
    PRETRAINED_BACKBONE,
)


class ImageNetNormalize(nn.Module):
    """
    Normalize images with ImageNet statistics.

    Input images are expected to be float tensors in [0, 1]
    with shape:

        [B, 3, H, W]

    This is needed because the ResNet18 backbone is pretrained on ImageNet.
    """

    def __init__(self):
        super().__init__()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        return (images - self.mean) / self.std


class ResNet18Backbone(nn.Module):
    """
    ResNet18 backbone without average pooling and classification head.

    For input images of size 640 x 640, the output feature map has spatial size:

        640 / 32 = 20

    So the output shape is:

        [B, 512, 20, 20]

    This matches GRID_SIZE = 20.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet18YOLOStudent(nn.Module):
    """
    ResNet18 single-stage student detector.

    Input:
        images: [B, 3, 640, 640], values in [0, 1]

    Output:
        predictions: [B, NUM_BOXES * (1 + 4 + NUM_CLASSES), 20, 20]

    For each grid cell and each box slot, the model predicts:

        1 objectness logit
        4 bbox raw values: [tx, ty, w, h]
        NUM_CLASSES class logits

    Bbox encoding:
        tx, ty = local offsets inside the grid cell
        w, h = normalized global image width/height
    """

    def __init__(
        self,
        head_channels=HEAD_CHANNELS,
        num_classes=NUM_CLASSES,
        num_boxes=NUM_BOXES,
        pretrained_backbone=PRETRAINED_BACKBONE,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_boxes = int(num_boxes)
        self.values_per_box = 1 + 4 + self.num_classes

        self.normalizer = ImageNetNormalize()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        out_channels = self.num_boxes * self.values_per_box

        self.head = nn.Sequential(
            nn.Conv2d(512, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, out_channels, kernel_size=1),
        )

    def forward(self, images):
        images = self.normalizer(images)
        features = self.backbone(images)
        predictions = self.head(features)

        return predictions


def count_trainable_parameters(model):
    """
    Count trainable model parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model):
    """
    Count all model parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def build_student_model(
    num_classes=NUM_CLASSES,
    num_boxes=NUM_BOXES,
    head_channels=HEAD_CHANNELS,
    pretrained_backbone=PRETRAINED_BACKBONE,
):
    
    model = ResNet18YOLOStudent(
        num_classes=num_classes,
        num_boxes=num_boxes,
        head_channels=head_channels,
        pretrained_backbone=pretrained_backbone,
    )

    return model


def check_model_output_shape(device=None):
  

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_student_model().to(device)
    model.eval()

    expected_channels = NUM_BOXES * (1 + 4 + NUM_CLASSES)

    dummy_images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.no_grad():
        predictions = model(dummy_images)

    expected_shape = (2, expected_channels, GRID_SIZE, GRID_SIZE)

    if predictions.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected prediction shape: got {tuple(predictions.shape)}, "
            f"expected {expected_shape}"
        )

    print("Dummy images:", tuple(dummy_images.shape))
    print("Predictions:", tuple(predictions.shape))
    print("Expected channels:", expected_channels)
    print("Total parameters:", count_total_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))

    return model