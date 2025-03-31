# feature_engineering.py

import torch
from transformers import BertModel
import time

class FeatureEngineering:

    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def generate_embeddings(self, encodings, batch_size=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        all_embeddings = []
        total_batches = (len(encodings['input_ids']) + batch_size - 1) // batch_size

        start_time = time.time()

        for batch_num, i in enumerate(range(0, len(encodings['input_ids']), batch_size)):
            batch = {key: val[i:i + batch_size].to(device) for key, val in encodings.items()}

            batch_start = time.time()

            with torch.no_grad():
                outputs = self.model(**batch)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(batch_embeddings)

            batch_time = time.time() - batch_start
            elapsed_time = time.time() - start_time
            avg_batch_time = elapsed_time / (batch_num + 1)
            remaining_batches = total_batches - (batch_num + 1)
            estimated_remaining = avg_batch_time * remaining_batches
            print(f"Processed batch {batch_num + 1}/{total_batches} - "
                  f"Time per batch: {batch_time:.2f}s - "
                  f"Estimated time remaining: {estimated_remaining:.2f}s")

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings
