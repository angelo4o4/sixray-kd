# SIXray knowledge distillation

Master's research project on object detection in baggage X-ray images. We study whether features from an RT-DETRv2 teacher can help train a smaller ResNet18 student on the supplied SIXray subset, using SIXray-D annotations.

## Models and experiments

- **Teacher:** we use Hugging Face RT-DETRv2, `PekingU/rtdetr_v2_r50vd`, for baseline training and a second experiment with weighted sampling.
- **Student:** we use an ImageNet-pretrained ResNet18 backbone with a YOLO-style detection head. A 640 × 640 RGB image produces a 20 × 20 grid with two box slots per cell and five classes: gun, knife, wrench, pliers, scissors.
- **Feature distillation:** we freeze the teacher backbone and compare its 2048-channel features with the student's 512-channel features. A training-only 1 × 1 convolution maps the student features to 2048 channels. We project ground-truth boxes onto the feature grid and mark the union of intersecting cells as foreground. We compute squared distances between channel-normalized features, then average foreground and background separately per image. KD loss is foreground loss + 0.1 × background loss, averaged over the batch. Negative images have no foreground term and contribute 0.1 × background loss.

The student detection loss combines objectness BCE, 10 × box Smooth L1, and class cross entropy. For the matched comparison, we add KD with weight 1 and train both students from identical saved initial weights and the same shuffle seed. We use AdamW, a cosine schedule, and 50 epochs.

## Repository

```text
notebooks/
  01_dataset_deduplication.ipynb          # exploratory data preparation
  03_teacher_training_baseline.ipynb
  04_teacher_training_finetuned.ipynb
  05_teacher_student_inference.ipynb     # shared validation comparison
  06_student_feature_kd.ipynb            # matched supervised/KD comparison
  experiments/00_sanity_check.ipynb
src/                                    # teacher, data, evaluation, checkpoints
students/anna_student_resnet18/
  notebooks/resnet18_student.ipynb       # original supervised experiment
  scripts/                              # training, evaluation, matched KD
  src/sixray_student/                    # student implementation
scripts/setup_colab.sh
docs/                                   # dataset, reproduction, experiment notes
```

We keep the teacher and student implementations separate. Shared inference code uses both packages.

## Running the experiments

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

The full dataset, saved split, model checkpoints, and initial student weights are stored separately. The notebooks include qualitative dataset examples and detection visualizations. See [dataset format](docs/dataset_format.md) and [reproduction instructions](docs/reproduction.md) for the required files and commands.

Teacher training notebooks use Kaggle, and the original student notebook uses Colab. Notebooks 05/06 and the multi-seed runner use `artifacts/` under the repository root for data, checkpoints, and results. This directory is ignored by Git. Set the paths for your environment. The deduplication notebook explores data preparation but does not export the training dataset. Use the original annotations and saved split to reproduce the experiments.

## Results

The tables below report our saved notebook results. Scores are on a 0–1 scale; mAP averages IoU thresholds 0.50–0.95.

In [notebook 05](notebooks/05_teacher_student_inference.ipynb), we evaluate all three models on the same validation split:

| Model | mAP | AP50 | AP75 |
|---|---:|---:|---:|
| Baseline teacher | 0.7003 | 0.8330 | 0.7386 |
| Oversampling teacher | 0.7449 | 0.8662 | 0.7920 |
| Supervised student | 0.3556 | 0.6797 | 0.3204 |

In [notebook 06](notebooks/06_student_feature_kd.ipynb), we compare matched students with seed 42 and select each checkpoint by validation mAP:

| Student | Selected epoch | mAP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| Supervised control | 46 | 0.3813 | 0.6655 | 0.3859 |
| Feature KD | 45 | 0.4043 | 0.6882 | 0.4075 |

The student notebook also includes validation and test results. The multi-seed script needs additional runs stored outside the repository; no multi-seed summary is included here.

For notebooks 05/06, start the kernel in the repository root or the main `notebooks/` directory. Setup cells that clone remote branches may use a different code version from your local copy. See [running instructions](docs/reproduction.md) for details.

The [experiment notes](docs/experiment_notes.md) explain the teacher checkpoint issue, the difference between the baseline augmentation notes and current code, and differences between evaluation settings.
