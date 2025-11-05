"""Classifier processing stage - ML-based sentence quality assessment."""

import os
from typing import Optional, List

import pandas as pd

from ..config import LOG_DIR, DROP_THRESHOLD, KEEP_THRESHOLD
from ..models.classifier import load_tiny_classifier, classify_batch, route_by_score
from ..utils.io_utils import ensure_output_dir
from ..utils.logging import ensure_logdir, reset_log, write_jsonl
from ..utils.data_utils import batched
from .base import stage_path


def run_stage_classifier(args, input_path: Optional[str] = None) -> str:
    """Run the classifier processing stage.
    
    Args:
        args: Argument namespace with processing configuration
        input_path: Optional path to input file (defaults to heuristics output)
        
    Returns:
        Path to the output file
    """
    if not getattr(args, 'use_classifier', False) and input_path is None:
        raise RuntimeError("Classifier stage requested without enabling --use_classifier.")
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)
    reset_log("02_classifier.jsonl")

    source_path = input_path or stage_path(args, "heuristics")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Heuristics output not found at {source_path}. Run stage 'heuristics' first.")

    df = pd.read_parquet(source_path)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)

    if getattr(args, 'test_limit', None):
        original_len = len(df)
        limit = min(args.test_limit, len(df))
        df = df.head(limit).copy()
        print(f"[classifier][TEST] Using {len(df)} of {original_len} sentences.")

    # Ensure sentence column exists
    if "sentence" not in df.columns:
        raise ValueError("Input must have 'sentence' column. Run heuristics stage first.")

    df["sentence"] = df["sentence"].fillna("").astype(str)
    
    # Initialize classifier columns
    df["score_classifier"] = None
    df["keep"] = True  # Default: keep all
    df["confidence"] = None
    df["decision_source"] = "classifier_keep"
    df["llm_pending"] = False

    # Score all sentences
    clf = load_tiny_classifier(getattr(args, 'classifier_model', 'typeform/distilbert-base-uncased-mnli'))
    texts = df["sentence"].tolist()
    scores: List[float] = []
    
    print(f"[classifier] Scoring {len(texts):,} sentences...")
    batch_size = getattr(args, 'classifier_batch_size', 64)
    for chunk in batched(texts, batch_size):
        scores.extend(classify_batch(clf, chunk, batch_size))

    df["score_classifier"] = scores
    df["confidence"] = scores

    # Route by score
    keep_pos, drop_pos, gray_pos = route_by_score(scores)
    
    # Update decisions
    df.loc[df.index[keep_pos], "keep"] = True
    df.loc[df.index[keep_pos], "decision_source"] = "classifier_keep"

    df.loc[df.index[drop_pos], "keep"] = False
    df.loc[df.index[drop_pos], "decision_source"] = "classifier_drop"

    df.loc[df.index[gray_pos], "keep"] = False
    df.loc[df.index[gray_pos], "decision_source"] = "classifier_gray"
    df.loc[df.index[gray_pos], "llm_pending"] = True

    print(f"[classifier] Gray zone (dubious, needs LLM review): {len(gray_pos):,} sentences")
    print(f"[classifier] Score range for gray zone: {DROP_THRESHOLD:.2f} - {KEEP_THRESHOLD:.2f}")

    # Log samples
    cls_logs = []
    sample_indices = list(keep_pos[:10]) + list(drop_pos[:10]) + list(gray_pos[:10])
    for pos in sample_indices:
        if pos < len(df):
            row = df.iloc[pos]
            cls_logs.append({
                "stage": "classifier",
                "row_id": int(row["row_id"]),
                "split": str(row["split"]),
                "source_idx": int(row["source_idx"]),
                "title": str(row["title"]),
                "sentence": row["sentence"],
                "score": float(row["score_classifier"]),
                "bucket": row["decision_source"]
            })
    
    if cls_logs:
        write_jsonl(os.path.join(LOG_DIR, "02_classifier.jsonl"), cls_logs)

    print(f"[classifier] Triage → keep={len(keep_pos):,} drop={len(drop_pos):,} gray={len(gray_pos):,}")

    out_path = stage_path(args, "classifier")
    df.to_parquet(out_path, index=False)
    print(f"[classifier] Saved {out_path}")
    return out_path
