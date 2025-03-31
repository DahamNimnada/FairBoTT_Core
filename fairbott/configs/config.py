# config.py

# Dataset paths
ROOT_DATA_PATH = "/Users/dahamnimnada/development/projects/FairBoTTDataset/"
DATASET_CSV = ROOT_DATA_PATH + "fairbott_final_cleaned_dataset.csv"

# Model paths
MODEL_PATH = ROOT_DATA_PATH + "fairbott_bias_model_v1_detection.pth"
TRAIN_EMBEDDINGS_PATH = ROOT_DATA_PATH + "train_embeddings.pt"
VAL_EMBEDDINGS_PATH = ROOT_DATA_PATH + "val_embeddings.pt"
TRAIN_LABELS_PATH = ROOT_DATA_PATH + "train_labels.pt"
VAL_LABELS_PATH = ROOT_DATA_PATH + "val_labels.pt"

TOXIC_BERT_MODEL_PATH = "/Users/dahamnimnada/development/projects/FairBoTT Model History/Bias detection model/toxic_bert"