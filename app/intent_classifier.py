# app/intent_classifier.py
from transformers import pipeline

# Load the zero-shot classifier once (caches the model)
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1  # -1 for CPU, 0 for GPU if available
)

def classify_intent(message: str) -> str:
    """
    Classify user message into one of: greeting, off_topic, identity, meeting.
    """
    candidate_labels = ["greeting", "off_topic", "identity", "meeting"]
    result = classifier(message, candidate_labels)
    # Return the label with highest confidence
    return result["labels"][0]