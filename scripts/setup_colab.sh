#!/bin/bash
# Install Colab dependencies.
pip install -q -r /content/sixray-kd/requirements.txt

# Extract the dataset.
DATASET_NAME=${1:-"subset_clean.zip"}
DATASET_ZIP_PATH="/content/drive/MyDrive/DatasetAPAI/${DATASET_NAME}"
DEST="/content/data"
mkdir -p $DEST
unzip -q $DATASET_ZIP_PATH -d $DEST
echo "Dataset '${DATASET_NAME}' extracted to $DEST"