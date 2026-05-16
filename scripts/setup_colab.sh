#!/bin/bash
# Installing dependencies to be used in Colab
pip install -q -r /content/sixray-kd/requirements.txt

# Dataset extraction
DATASET_NAME=${1:-"subset.zip"}
DATASET_ZIP_PATH="/content/drive/MyDrive/DatasetAPAI/${DATASET_NAME}"
DEST="/content/data"
mkdir -p $DEST
unzip -q $DATASET_ZIP_PATH -d $DEST
echo "Dataset '${DATASET_NAME}' extracted to $DEST"