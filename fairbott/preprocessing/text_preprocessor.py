from transformers import BertTokenizer

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, texts, max_length=256):  # Increased max_length for richer samples
        print(f"Tokenizing {len(texts)} records...")  # Optional log
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )