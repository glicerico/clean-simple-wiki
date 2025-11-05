"""Finalize stage - output clean sentences ready for semantic parsing."""

import os
from typing import List, Tuple

import pandas as pd
import orjson

from ..config import LOG_DIR
from ..utils.io_utils import ensure_output_dir
from ..utils.logging import ensure_logdir, reset_log, write_jsonl
from .base import stage_path


def run_stage_finalize(args) -> str:
    """Run the finalize processing stage.
    
    Args:
        args: Argument namespace with processing configuration
        
    Returns:
        Path to the output parquet file
    """
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)
    reset_log("04_final_decisions.jsonl")

    # Try to find the latest stage output
    candidates = [
        ("llm", stage_path(args, "llm")),
        ("classifier", stage_path(args, "classifier")),
        ("heuristics", stage_path(args, "heuristics")),
    ]
    source_stage = None
    source_path = None
    for stage_name, path in candidates:
        if os.path.exists(path):
            source_stage = stage_name
            source_path = path
            break
    if source_path is None:
        raise FileNotFoundError("No stage outputs found. Run at least the heuristics stage before finalizing.")

    df = pd.read_parquet(source_path)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)

    if getattr(args, 'test_limit', None):
        original_len = len(df)
        limit = min(args.test_limit, len(df))
        df = df.head(limit).copy()
        print(f"[finalize][TEST] Using {len(df)} of {original_len} sentences.")

    # Ensure required columns exist
    df["sentence"] = df["sentence"].fillna("").astype(str) if "sentence" in df.columns else df["chunk_clean"].fillna("").astype(str)
    
    # If no keep/decision_source columns, assume all are kept from heuristics
    if "keep" not in df.columns:
        df["keep"] = True
        df["decision_source"] = source_stage

    # Filter to kept sentences only
    kept = df[df["keep"]].copy()
    
    # Select minimal columns for semantic parsing
    output_cols = ["row_id", "title", "source_idx", "split", "sentence"]
    if "sentence_idx" in kept.columns:
        output_cols.insert(3, "sentence_idx")
    if "decision_source" in kept.columns:
        output_cols.append("decision_source")
    
    kept = kept[[col for col in output_cols if col in kept.columns]]

    # Log final decisions (sample)
    final_logs = []
    for idx in kept.index[:100]:  # Log first 100 for review
        log_entry = {
            "stage": "final",
            "row_id": int(kept.loc[idx, "row_id"]),
            "source_stage": source_stage,
            "split": str(kept.loc[idx, "split"]),
            "source_idx": int(kept.loc[idx, "source_idx"]),
            "title": str(kept.loc[idx, "title"]),
            "sentence": str(kept.loc[idx, "sentence"]),
        }
        if "decision_source" in kept.columns:
            log_entry["decision_source"] = str(kept.loc[idx, "decision_source"])
        final_logs.append(log_entry)
    
    if final_logs:
        write_jsonl(os.path.join(LOG_DIR, "04_final_decisions.jsonl"), final_logs)

    # Write final outputs
    jsonl_path = getattr(args, 'output_jsonl', None) or os.path.join(args.output_dir, f"{args.out_prefix}.jsonl")
    parquet_path = getattr(args, 'output_parquet', None) or os.path.join(args.output_dir, f"{args.out_prefix}.parquet")
    ensure_output_dir(os.path.dirname(jsonl_path) or ".")
    ensure_output_dir(os.path.dirname(parquet_path) or ".")

    with open(jsonl_path, "wb") as f:
        for _, row in kept.iterrows():
            rec = {
                "row_id": int(row["row_id"]),
                "title": row["title"],
                "source_idx": int(row["source_idx"]),
                "split": row["split"],
                "sentence": row["sentence"],
            }
            if "sentence_idx" in row:
                rec["sentence_idx"] = int(row["sentence_idx"])
            if "decision_source" in row:
                rec["decision_source"] = row["decision_source"]
            f.write(orjson.dumps(rec))
            f.write(b"\n")
    kept.to_parquet(parquet_path, index=False)

    num_sources = kept["source_idx"].nunique() if "source_idx" in kept.columns else 0
    print(f"[finalize] Output {len(kept):,} sentences from {num_sources:,} source documents.")
    print(f"[finalize] Ready for semantic parsing!")
    print(f"[finalize] Artifacts: {jsonl_path}, {parquet_path}")
    return parquet_path
