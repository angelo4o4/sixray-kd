import random
import warnings
from collections import Counter


def _pos_neg_indices(dataset) -> tuple[list[int], list[int]]:
    positive_idx, negative_idx = [], []
    for idx in range(len(dataset)):
        if len(dataset.annotations_per_image[idx]) > 0:
            positive_idx.append(idx)
        else:
            negative_idx.append(idx)
    return positive_idx, negative_idx


def balanced_subset_indices(dataset, num_per_class, seed):
    """Balanced mini-subset: up to num_per_class positive and negative images."""
    positive_idx, negative_idx = _pos_neg_indices(dataset)

    rng = random.Random(seed)
    train_indices = (
        rng.sample(negative_idx, min(num_per_class, len(negative_idx)))
        + rng.sample(positive_idx, min(num_per_class, len(positive_idx)))
    )
    rng.shuffle(train_indices)
    return train_indices, len(positive_idx), len(negative_idx)


def create_train_val_split(dataset, val_pos, val_neg, train_total, seed):
    """Hold out fixed pos/neg val images; fill train up to train_total from the rest."""
    positive_idx, negative_idx = _pos_neg_indices(dataset)

    if val_pos > len(positive_idx):
        raise ValueError(f"val_pos={val_pos} exceeds available positives ({len(positive_idx)})")
    if val_neg > len(negative_idx):
        raise ValueError(f"val_neg={val_neg} exceeds available negatives ({len(negative_idx)})")

    rng = random.Random(seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    val_indices = positive_idx[:val_pos] + negative_idx[:val_neg]
    avail_train_pos = positive_idx[val_pos:]

    if train_total < len(avail_train_pos):
        num_train_pos = train_total // 2
    else:
        num_train_pos = len(avail_train_pos)
        
    train_pos = avail_train_pos[:num_train_pos]
    need_neg = train_total - len(train_pos)
    avail_train_neg_count = len(negative_idx) - val_neg

    if need_neg > avail_train_neg_count:
        warnings.warn(
            f"Requested {need_neg} train negatives but only {avail_train_neg_count} available; "
            f"train set will have {len(train_pos) + avail_train_neg_count} images instead of {train_total}.",
            stacklevel=2,
        )
        need_neg = avail_train_neg_count

    train_indices = train_pos + negative_idx[val_neg : val_neg + need_neg]
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices, len(positive_idx), len(negative_idx)


def class_distribution(indices, dataset, id2label):
    counter = Counter()
    for idx in indices:
        for ann in dataset.annotations_per_image[idx]:
            counter[id2label[ann["category_id"]]] += 1
    return counter


def count_pos_neg(indices, dataset):
    pos = sum(1 for i in indices if len(dataset.annotations_per_image[i]) > 0)
    neg = len(indices) - pos
    return pos, neg


def print_pos_neg_balance(name, indices, dataset):
    pos, neg = count_pos_neg(indices, dataset)
    total = pos + neg
    if total == 0:
        print(f"{name}: empty subset")
        return
    print(
        f"{name}: {100 * pos / total:.1f}% positives ({pos}) | "
        f"{100 * neg / total:.1f}% negatives ({neg})"
    )


def get_stats(dataset, name: str):
    total_images = len(dataset.images)
    positives = sum(1 for anns in dataset.annotations_per_image if len(anns) > 0)
    perc = (positives / total_images) * 100 if total_images else 0.0
    print(f"{name} Set: {positives}/{total_images} positive images ({perc:.2f}%)")
    return positives, total_images, perc


