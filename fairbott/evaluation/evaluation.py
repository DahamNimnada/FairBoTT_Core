import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from fairbott.model.bias_detection_model import BiasDetectionModel
from fairbott.configs.config import MODEL_PATH, VAL_EMBEDDINGS_PATH, VAL_LABELS_PATH
from fairbott.data.keywords.bias_keywords import BIAS_CATEGORIES

class Evaluation:
    def __init__(self, model, val_embeddings, val_labels):
        self.model = model
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels

    def calculate_metrics(self):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(len(self.val_embeddings['input_ids'])):
                input_ids = self.val_embeddings['input_ids'][i].unsqueeze(0)
                attention_mask = self.val_embeddings['attention_mask'][i].unsqueeze(0)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
                predictions.append(prediction)

        accuracy = accuracy_score(self.val_labels, predictions)
        precision = precision_score(self.val_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.val_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.val_labels, predictions, average='weighted', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return predictions

    def plot_confusion_matrix(self, predictions):
        cm = confusion_matrix(self.val_labels, predictions)
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=BIAS_CATEGORIES, yticklabels=BIAS_CATEGORIES)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Bias Type Classification")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()


def evaluate_model():
    # Load model
    model = BiasDetectionModel(num_labels=19)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load embeddings and labels
    val_embeddings = torch.load(VAL_EMBEDDINGS_PATH)
    val_labels = torch.load(VAL_LABELS_PATH)

    # Run evaluation
    evaluator = Evaluation(model, val_embeddings, val_labels)
    predictions = evaluator.calculate_metrics()
    evaluator.plot_confusion_matrix(predictions)


if __name__ == '__main__':
    evaluate_model()