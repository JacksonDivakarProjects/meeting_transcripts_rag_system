# app/intent_classifier.py
from transformers import pipeline

# facebook/bart-large-mnli is the standard, publicly available zero-shot model.
# typeform/distilbert-base-uncased-mnli was removed from the Hub.
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1,  # -1 = CPU; change to 0 for GPU
        )
    return _classifier


CANDIDATE_LABELS = ["greeting", "off_topic", "identity", "meeting"]


def classify_intent(message: str) -> str:
    """
    Classify the user message into one of:
      greeting | off_topic | identity | meeting
    Returns 'meeting' on any error so the RAG pipeline always gets a chance.
    """
    try:
        result = _get_classifier()(message, CANDIDATE_LABELS)
        return result["labels"][0]
    except Exception as e:
        print(f"Intent classifier error: {e} — defaulting to 'meeting'")
        return "meeting"
