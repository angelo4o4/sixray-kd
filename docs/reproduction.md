# Running the experiments

Run commands from the repository root. We used Colab, Kaggle, and Windows for different experiments, so each notebook has its own path settings. For notebooks 05 and 06, start the kernel in the repository root or the main `notebooks/` directory.

## Environment

```bash
python -m pip install -r requirements.txt
```

Requirements are mostly unpinned; use the original package versions when available. Pretrained weights may need to be downloaded from Hugging Face or torchvision. The multi-seed runner requires CUDA.

Some notebooks need packages not listed explicitly in the requirements: pandas for tables, scikit-learn for the deduplication split, ImageHash for perceptual hashing, and ipywidgets for the sanity-check slider. The deduplication notebook installs ImageHash. Run Colab and Kaggle-specific cells on those platforms; shell and magic commands need an IPython notebook environment.

## Required files

| Files | Used by | Details |
|---|---|---|
| Image archive and annotation JSON | All main experiments | Use the original images and `images` ordering. See [dataset format](dataset_format.md). Later notebooks use `subset_clean.zip`. |
| `split_seed42_train10500_val300pos1200neg.json` | Student scripts and notebooks 05/06 | Train/validation/test index lists matched to the annotation files. |
| Teacher checkpoint directories | Notebook 05; oversampling teacher also used for KD | Hugging Face model/config/processor files in `01_rtdetr_teacher_baseline_angelo_best` and `02_rtdetr_teacher_oversampling_angelo_best`. |
| Student checkpoint | Notebook 05 and standalone evaluation | `student_resnet18_yolo2_local_offsets_best_map_50.pth`, with matching architecture and class metadata. |
| Initial student weights | Notebook 06 and multi-seed preparation | The same `initial_student.pth` used for the seed-42 comparison. |
| Seed-42 control/KD run | Notebook 06 results and multi-seed runner | Best checkpoints and training histories; the runner also reads `kd/config.json`. |
| Resume files | Continuing a previous run | Teacher epoch/optimizer/scheduler files, or the student last-checkpoint dictionary. |

The deduplication notebook also needs `subset.zip` and `SIXray-D.zip` with XML annotations. It does not export the files used by later training, so these must be obtained separately.

Teacher checkpoints can include `model_full_weights.pt` for weights, `training_meta.json` for epoch and metric information, and `training_state.pt` for optimizer/scheduler state. The loader supports older weight keys and strict Hugging Face loading. The student loader accepts checkpoint dictionaries and raw state dictionaries.

## Workflow

### Teacher training

Use notebooks 03 and 04 with the original Kaggle dataset/checkpoint paths. Their setup cells clone the repository; some select the `angelo` branch. This can load a different source version from your local checkout. See [experiment notes](experiment_notes.md) for the checkpoint and augmentation issues.

### Student training and evaluation

Set paths in `students/anna_student_resnet18/src/sixray_student/config.py`. Configure W&B in the same package's `wandb_config.py`, or disable it there.

```bash
python students/anna_student_resnet18/scripts/train_student.py
python students/anna_student_resnet18/scripts/evaluate_student.py
```

Both scripts build train, validation, and test datasets, so all three splits must be available. Training can resume from the configured last checkpoint and stops at the total epoch count in the configuration.

### Model comparisons

Notebooks 05 and 06 set `ARTIFACT_ROOT = REPO_ROOT / "artifacts"`. Put the required files under this directory, or change that one path to an existing directory. Keep the layout below; `artifacts/` is ignored by Git.

```text
artifacts/
  DatasetAPAI/
    subset_clean.zip
    subset_clean/train.json
    subset_clean/train/images/...
    SIXray_Project/
      splits/split_seed42_train10500_val300pos1200neg.json
      checkpoints_angelo/...
      student_checkpoints_anna/...
  experiments/
    feature_kd_20261016T214646677055Z/initial_student.pth
    matched_20261016T221317947683Z/...
```

In notebook 05, we evaluate saved teacher/student checkpoints on the validation split. Metrics and settings are saved under `artifacts/evaluation_results/`.

In notebook 06, we train matched students from the same saved initial weights. `RUN_TRAINING = False` reads the existing results without training.

### Multiple seeds

The multi-seed script sets `ARTIFACTS = ROOT / "artifacts"` in `run_matched_feature_kd.py`, using the same layout as the comparison notebooks. It expects `experiments/feature_kd_20261016T214646677055Z/initial_student.pth` and the seed-42 run in `experiments/matched_20261016T221317947683Z`. Set these paths and the paths in the run configuration for your files.

```bash
python students/anna_student_resnet18/scripts/run_matched_feature_kd.py --prepare-only <new_output_directory>
python students/anna_student_resnet18/scripts/run_matched_feature_kd.py --run <new_output_directory>
```

Replace `<new_output_directory>` with a new path. Preparation creates initial weights for seeds 43–46 and checks that each control/KD pair starts identically. It needs CUDA and the seed-42 files. The run command trains both students for these four seeds and combines their results with the seed-42 histories.

### Colab setup

After mounting Drive, run:

```bash
bash scripts/setup_colab.sh subset_clean.zip
```

The helper expects the repository at `/content/sixray-kd`, reads the archive from `/content/drive/MyDrive/DatasetAPAI/`, and extracts into `/content/data`. The archive must have the layout expected by the notebook.
