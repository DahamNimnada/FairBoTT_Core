import torch
import torch.nn.functional as tf
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

CONFIDENCE_THRESHOLD = 0.65

print("Loading FairBoTT model and NLP components...")

# === Load FairBoTT model (binary classification) ===
fairbott_model = BiasDetectionModel(num_labels=2)
fairbott_model.load_state_dict(torch.load(MODEL_PATH))
fairbott_model.eval()

# === Tokenizers ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# === Fallback bias type matcher ===
matcher = PhraseMatcher(nlp.vocab)
for category, phrases in BIAS_CATEGORIES.items():
    patterns = [nlp.make_doc(phrase) for phrase in phrases]
    matcher.add(category, patterns)

# === Load Toxic-BERT ===
print("Loading local Toxic-BERT model...")
toxic_tokenizer = AutoTokenizer.from_pretrained(TOXIC_BERT_MODEL_PATH)
toxic_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_BERT_MODEL_PATH)
toxic_bias_detector = pipeline("text-classification", model=toxic_model, tokenizer=toxic_tokenizer)

# === Context downgrade logic ===
def downgrade_if_weak_context(sentence, predicted_type):
    doc = nlp(sentence)
    if len(doc) < 5:
        return "unknown"

    strong_entities = {
        "PERSON", "NORP", "ORG", "GPE", "LOC", "EVENT",
        "WORK_OF_ART", "LANGUAGE", "DATE", "TIME", "MONEY", "LAW", "FAC", "PRODUCT"
    }

    has_context = any(ent.label_ in strong_entities for ent in doc.ents)
    matches = matcher(doc)
    has_keyword_hit = bool(matches)

    if not has_context and not has_keyword_hit:
        return "unknown"
    return predicted_type

# === Fallback bias type using keyword matcher ===
def map_bias_category(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    detected = {nlp.vocab.strings[match_id] for match_id, _, _ in matches}
    return list(detected) if detected else ["unknown"]

# === FairBoTT Bias Detector (binary) ===
def detect_bias_fairbott(text):
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = fairbott_model(encoded_input["input_ids"], encoded_input["attention_mask"])
        logits = outputs.logits
        probs = tf.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    if prediction == 1:  # Biased
        fallback = map_bias_category(text)
        downgraded = downgrade_if_weak_context(text, fallback[0]) if fallback != ["unknown"] else "unknown"
        print("downgraded: ", downgraded)
        return "bias", confidence, downgraded
    else:
        return "neutral", confidence, "neutral"

# === Toxic-BERT Detector ===
def detect_bias_toxicbert(text):
    results = toxic_bias_detector(text)
    labels = [r["label"] for r in results if r["score"] > 0.5]
    return "bias" if labels else "neutral"

# === Final Comparison Logic ===
def compare_results(model_result, toxic_result, text, confidence, bias_type_hint):
    if model_result == "neutral" and toxic_result == "neutral":
        return {
            "final_decision": "neutral",
            "bias_type": "neutral"
        }

    if model_result == "neutral" and toxic_result == "bias":
        return {
            "final_decision": "bias",
            "bias_type": map_bias_category(text),
            "store_for_retraining": text
        }

    if model_result == "bias" and toxic_result == "neutral":
        if confidence >= CONFIDENCE_THRESHOLD and bias_type_hint != "unknown":
            return {
                "final_decision": "bias",
                "bias_type": bias_type_hint
            }
        return {
            "final_decision": "neutral",
            "bias_type": "neutral"
        }

    if model_result == "bias" and toxic_result == "bias":
        return {
            "final_decision": "bias",
            "bias_type": bias_type_hint if bias_type_hint != "unknown" else map_bias_category(text)
        }

    return {
        "final_decision": "neutral",
        "bias_type": "neutral"
    }

# === Main Detection Pipeline ===
def detect_bias_from_text(text):
    sentences = sent_tokenize(text)
    has_bias = False
    bias_types_found = set()
    sentence_results = []
    confidence_scores = []

    for idx, sentence in enumerate(sentences):
        clean_sentence = sentence.strip()
        print(f"\nAnalyzing: {clean_sentence}")
        model_result, confidence, inferred_type = detect_bias_fairbott(clean_sentence)
        toxic_result = detect_bias_toxicbert(clean_sentence)

        print(f"FairBoTT Result: {model_result} (Confidence: {confidence:.2f})")
        print(f"Toxic-BERT Result: {toxic_result}")

        decision = compare_results(model_result, toxic_result, clean_sentence, confidence, inferred_type)
        print(f"→ Decision: {decision}")

        confidence_scores.append(confidence)

        sentence_results.append({
            "index": idx,
            "text": clean_sentence,
            "decision": decision["final_decision"],
            "bias_type": decision["bias_type"],
            "confidence": round(confidence, 3)
        })

        if decision["final_decision"] == "bias":
            has_bias = True
            if isinstance(decision["bias_type"], list):
                bias_types_found.update(decision["bias_type"])
            else:
                bias_types_found.add(decision["bias_type"])

    average_confidence = round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else 0.0

    return {
        "final_decision": "bias" if has_bias else "neutral",
        "bias_types": list(bias_types_found) if has_bias else ["neutral"],
        "sentence_results": sentence_results,
        "average_confidence": average_confidence
    }

# === Manual Testing ===
if __name__ == "__main__":
    print("\nFairBoTT Bias Detection Console (Binary Model + Context-Aware Inference)\n")
    while True:
        paragraph = input("Enter a paragraph to test for bias (or press Ctrl+C to exit): ")
        result = detect_bias_from_text(paragraph)
        print("\n=== Final Result ===")
        print(result)