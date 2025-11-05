"""Analysis utilities for processing stages."""

from typing import List, Dict, Any

from ..config import LOG_DIR
from ..utils.logging import read_jsonl


def analyze_failed_batches(log_dir: str = LOG_DIR) -> List[Dict[str, Any]]:
    """Analyze failed batch logs to help debug LLM processing issues.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        List of failed batch records
    """
    import os
    
    failed_batches_path = os.path.join(log_dir, "llm_failed_batches.jsonl")
    
    if not os.path.exists(failed_batches_path):
        print(f"[analyze] No failed batches file found at {failed_batches_path}")
        return []
    
    failed_records = read_jsonl(failed_batches_path)
    if not failed_records:
        print(f"[analyze] No failed batch records found")
        return []
    
    print(f"[analyze] Found {len(failed_records)} failed sentences")
    
    # Group by common characteristics
    by_title = {}
    by_length = {"short": 0, "medium": 0, "long": 0}
    by_score = {"low": 0, "medium": 0, "high": 0}
    
    for rec in failed_records:
        # Group by title
        title = rec.get("title", "unknown")
        by_title[title] = by_title.get(title, 0) + 1
        
        # Group by sentence length
        sentence_len = len(rec.get("sentence", ""))
        if sentence_len < 50:
            by_length["short"] += 1
        elif sentence_len < 150:
            by_length["medium"] += 1
        else:
            by_length["long"] += 1
        
        # Group by classifier score
        score = rec.get("score_classifier", 0.0)
        if score < 0.45:
            by_score["low"] += 1
        elif score < 0.65:
            by_score["medium"] += 1
        else:
            by_score["high"] += 1
    
    print(f"[analyze] By sentence length: {by_length}")
    print(f"[analyze] By classifier score: {by_score}")
    print(f"[analyze] Top failing titles:")
    for title, count in sorted(by_title.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"[analyze]   {title}: {count} failures")
    
    # Show sample failed sentences
    print(f"[analyze] Sample failed sentences:")
    for i, rec in enumerate(failed_records[:3]):
        print(f"[analyze]   {i+1}. '{rec.get('sentence', '')[:80]}...' (score: {rec.get('score_classifier', 0.0):.2f})")
    
    return failed_records
