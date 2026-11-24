import random
import copy
from torchvision import transforms as T


class TrainTransform:
    """Image and COCO box augmentation before the Hugging Face processor."""

    def __init__(self, flip_p=0.5, vflip_p=0.5, brightness=0.2, contrast=0.2):
        self.flip_p = flip_p
        self.vflip_p = vflip_p
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast)

    def _clip_bbox(self, bbox, width, height):
        """Clip the box to the image boundaries."""
        x, y, w, h = bbox
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(width, x + w)
        y_max = min(height, y + h)
        return [x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min)]

    def __call__(self, image, target):
        width, height = image.size
        target = copy.deepcopy(target)
        # horizontal flip
        if random.random() < self.flip_p:
            image = T.functional.hflip(image)
            for ann in target["annotations"]:
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [width - x - w, y, w, h]

        # vertical flip
        if random.random() < self.vflip_p:
            image = T.functional.vflip(image)
            for ann in target["annotations"]:
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [x, height - y - h, w, h]

        # Clip boxes and drop boxes with no area.
        valid_annotations = []
        for ann in target["annotations"]:
            ann["bbox"] = self._clip_bbox(ann["bbox"], width, height)
            if ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                valid_annotations.append(ann)

        target["annotations"] = valid_annotations
        image = self.color_jitter(image)
        return image, target


def build_train_transforms(flip_p=0.5, vflip_p=0.5, brightness=0.2, contrast=0.2, enabled=True):
    if not enabled:
        return None
    return TrainTransform(flip_p=flip_p, vflip_p=vflip_p, brightness=brightness, contrast=contrast)
