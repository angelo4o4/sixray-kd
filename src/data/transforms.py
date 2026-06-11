import random

from torchvision import transforms as T


class TrainTransform:
    """PIL + COCO bbox augmentations applied before the HF processor."""

    def __init__(self, flip_p=0.5, brightness=0.2, contrast=0.2):
        self.flip_p = flip_p
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, image, target):
        if random.random() < self.flip_p:
            image = T.functional.hflip(image)
            width, _ = image.size
            for ann in target["annotations"]:
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [width - x - w, y, w, h]

        image = self.color_jitter(image)
        return image, target


def build_train_transforms(flip_p=0.5, brightness=0.2, contrast=0.2, enabled=True):
    if not enabled:
        return None
    return TrainTransform(flip_p=flip_p, brightness=brightness, contrast=contrast)
