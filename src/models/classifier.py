"""Text classification model for sentence quality assessment."""

from typing import List, Tuple

from ..config import KEEP_THRESHOLD, DROP_THRESHOLD

# Optional dependency handling
try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


def load_tiny_classifier(model_name: str = "typeform/distilbert-base-uncased-mnli"):
    """Load a zero-shot classification model.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Loaded classification pipeline
        
    Raises:
        RuntimeError: If transformers is not installed
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed.")
    return pipeline("zero-shot-classification", model=model_name, device=-1, truncation=True)


def classify_batch(clf, texts: List[str], batch_size: int) -> List[float]:
    """Classify a batch of texts for quality.
    
    Args:
        clf: Classification pipeline
        texts: List of texts to classify
        batch_size: Batch size for processing
        
    Returns:
        List of quality scores (0-1, higher is better)
    """
    labels = ["valuable", "not valuable"]
    preds = clf(
        texts,
        candidate_labels=labels,
        hypothesis_template="This chunk is {} for general knowledge.",
        multi_label=False,
        batch_size=batch_size
    )
    if isinstance(preds, dict):
        preds = [preds]
    scores = []
    for p in preds:
        d = {lab: sc for lab, sc in zip(p["labels"], p["scores"])}
        scores.append(float(d.get("valuable", 0.0)))
    return scores


def route_by_score(scores: List[float]) -> Tuple[List[int], List[int], List[int]]:
    """Route sentences by classification score into keep/drop/gray buckets.
    
    Args:
        scores: List of classification scores
        
    Returns:
        Tuple of (keep_indices, drop_indices, gray_indices)
    """
    keep, drop, gray = [], [], []
    for i, s in enumerate(scores):
        if s >= KEEP_THRESHOLD:
            keep.append(i)
        elif s <= DROP_THRESHOLD:
            drop.append(i)
        else:
            gray.append(i)
    return keep, drop, gray
