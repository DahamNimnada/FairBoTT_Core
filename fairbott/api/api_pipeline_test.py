import torch
import spacy
from nltk import sent_tokenize
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
)
from spacy.matcher import PhraseMatcher

from fairbott.model.bias_detection_model import BiasDetectionModel
from fairbott.data.keywords.bias_keywords import BIAS_CATEGORIES
from fairbott.configs.config import MODEL_PATH, TOXIC_BERT_MODEL_PATH

print("Loading FairBoTT model and NLP components...")

# === Load FairBoTT model ===
fairbott_model = BiasDetectionModel(num_labels=19)
fairbott_model.load_state_dict(torch.load(MODEL_PATH))
fairbott_model.eval()

# === Tokenizers ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# === Bias types used in training ===
bias_types = [
    "ability", "body_type", "characteristics", "cultural", "disabled", "gender",
    "nationality", "neutral", "political_ideologies", "profession", "race",
    "race_ethnicity", "religion", "social", "socioeconomic_class", "unknown",
    "victim", "bias_general", "other"
]

# === Keyword matcher for fallback bias detection ===
matcher = PhraseMatcher(nlp.vocab)
for category, phrases in BIAS_CATEGORIES.items():
    patterns = [nlp.make_doc(phrase) for phrase in phrases]
    matcher.add(category, patterns)

# === Load Toxic-BERT (local path) ===
print("Loading local Toxic-BERT model...")
toxic_tokenizer = AutoTokenizer.from_pretrained(TOXIC_BERT_MODEL_PATH)
toxic_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_BERT_MODEL_PATH)
toxic_bias_detector = pipeline("text-classification", model=toxic_model, tokenizer=toxic_tokenizer)

# === Context-aware downgrade filter ===
def downgrade_if_weak_context(sentence, predicted_type):
    doc = nlp(sentence)

    # Step 1: Rule out very short sentences
    if len(doc) < 5:
        return "unknown"

    # Step 2: Check for meaningful named entities
    strong_entities = {
        "PERSON", "NORP", "ORG", "GPE", "LOC", "EVENT",
        "WORK_OF_ART", "LANGUAGE", "DATE", "TIME", "MONEY", "LAW", "FAC", "PRODUCT"
    }

    has_context = any(ent.label_ in strong_entities for ent in doc.ents)

    # Step 3: Check for keyword-based category match
    matches = matcher(doc)
    has_keyword_hit = bool(matches)

    # Step 4: Apply downgrade if no strong context or keyword match
    if not has_context and not has_keyword_hit:
        return "unknown"

    return predicted_type

# === Fallback matcher bias type detection ===
def map_bias_category(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    detected = {nlp.vocab.strings[match_id] for match_id, _, _ in matches}
    return list(detected) if detected else ["unknown"]

# === Stage 1: FairBoTT prediction ===
def detect_bias_fairbott(text):
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = fairbott_model(encoded_input["input_ids"], encoded_input["attention_mask"])
        prediction = torch.argmax(outputs.logits, dim=1).item()
    raw_type = bias_types[prediction]
    final_type = downgrade_if_weak_context(text, raw_type)
    return final_type

# === Stage 2: Toxic-BERT prediction ===
def detect_bias_toxicbert(text):
    results = toxic_bias_detector(text)
    labels = [r["label"] for r in results if r["score"] > 0.5]
    return "bias" if labels else "neutral"

# === Decision logic ===
def compare_results(model_result, toxic_result, text):
    if model_result != "neutral" and toxic_result == "bias":
        return {"final_decision": "bias", "bias_type": model_result}
    elif model_result != "neutral" and toxic_result == "neutral":
        if model_result not in ["unknown", "other"]:
            return {"final_decision": "bias", "bias_type": model_result}
        else:
            return {"final_decision": "neutral", "bias_type": "neutral"}
    elif model_result == "neutral" and toxic_result == "bias":
        return {
            "final_decision": "bias",
            "bias_type": map_bias_category(text),
            "store_for_retraining": text
        }
    else:
        return {"final_decision": "neutral", "bias_type": "neutral"}

# === Main pipeline ===
def detect_bias_from_text(text):
    sentences = sent_tokenize(text)
    bias_types_found = set()
    has_bias = False
    sentence_results = []

    for sentence in sentences:
        print(f"\nAnalyzing: {sentence}")
        model_result = detect_bias_fairbott(sentence)
        toxic_result = detect_bias_toxicbert(sentence)

        print(f"FairBoTT Result: {model_result}")
        print(f"Toxic-BERT Result: {toxic_result}")

        decision = compare_results(model_result, toxic_result, sentence)
        print(f"→ Decision for sentence: {decision}")

        sentence_results.append({
            "text": sentence,
            "decision": decision["final_decision"],
            "bias_type": decision["bias_type"]
        })

        if decision["final_decision"] == "bias":
            has_bias = True
            if isinstance(decision["bias_type"], list):
                bias_types_found.update(decision["bias_type"])
            else:
                bias_types_found.add(decision["bias_type"])

    return {
        "final_decision": "bias" if has_bias else "neutral",
        "bias_types": list(bias_types_found) if has_bias else ["neutral"],
        "sentence_results": sentence_results
    }

# === CLI for manual testing ===
if __name__ == "__main__":
    print("\nFairBoTT Bias Detection Console (Toxic-BERT Enabled)\n")
    while True:
        paragraph = input("Enter a paragraph to test for bias (or press Ctrl+C to exit): ")
        result = detect_bias_from_text(paragraph)
        print("\n=== Final Result ===")
        print(result)