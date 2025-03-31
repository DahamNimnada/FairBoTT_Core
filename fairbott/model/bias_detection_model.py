import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
import time

class BiasDetectionModel(nn.Module):

    def __init__(self, num_labels=19):
        super(BiasDetectionModel, self).__init__()
        # Load pre-trained BERT for sequence classification
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def train_model(self, train_embeddings, train_labels,
                    val_embeddings, val_labels, epochs=5, batch_size=16, lr=2e-5):
        train_dataset = EmbeddingDataset(train_embeddings, train_labels)
        val_dataset = EmbeddingDataset(val_embeddings, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = Adam(self.parameters(), lr=lr)

        # Compute class weights to handle class imbalance
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=self.model.classifier.out_features)
        class_counts[class_counts == 0] = 1  # Prevent division by zero
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()

        loss_fn = CrossEntropyLoss(weight=class_weights)

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            start_time = time.time()

            self.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                optimizer.zero_grad()
                outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            print(f"[Training] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

            # Validation
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']

                    outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / total
            elapsed = time.time() - start_time
            print(f"[Validation] Accuracy: {val_accuracy:.4f} | Time: {elapsed:.2f}s")

    def save_model(self, path="/path/to/save/model.pth"):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")


class EmbeddingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)