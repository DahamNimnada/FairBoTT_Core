import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from fairbott.data.data_preperation import DataPreparation
from fairbott.model.bias_detection_model import BiasDetectionModel
from fairbott.configs.config import (
    DATASET_CSV, MODEL_PATH, TRAIN_EMBEDDINGS_PATH, VAL_EMBEDDINGS_PATH, VAL_LABELS_PATH, TRAIN_LABELS_PATH
)

def main():
    # Load and preprocess data
    print("Loading dataset...")
    data_prep = DataPreparation(DATASET_CSV, max_rows=100000)
    data_prep.load_data()
    data_prep.drop_missing_values()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get the posts (texts) and labels
    texts, labels = data_prep.get_data()

    # Tokenize the texts using BERT tokenizer
    print("Tokenizing data...")
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)

    # Split data into training and validation sets
    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
        encodings['input_ids'], encodings['attention_mask'], labels, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(train_labels)}")
    print(f"Validation set size: {len(val_labels)}")

    # Save the embeddings and labels
    torch.save({'input_ids': train_input_ids, 'attention_mask': train_attention_mask}, TRAIN_EMBEDDINGS_PATH)
    torch.save({'input_ids': val_input_ids, 'attention_mask': val_attention_mask}, VAL_EMBEDDINGS_PATH)
    torch.save(train_labels, TRAIN_LABELS_PATH)
    torch.save(val_labels, VAL_LABELS_PATH)

    # Initialize model for binary classification
    bias_model = BiasDetectionModel(num_labels=2)

    print(f"Unique labels in training set: {set(train_labels)}")

    # Train the model
    bias_model.train_model(
        train_embeddings={'input_ids': train_input_ids, 'attention_mask': train_attention_mask},
        train_labels=train_labels,
        val_embeddings={'input_ids': val_input_ids, 'attention_mask': val_attention_mask},
        val_labels=val_labels
    )

    # Save the trained model
    bias_model.save_model(path=MODEL_PATH)

    print("Training complete. Model and embeddings saved.")

if __name__ == '__main__':
    main()