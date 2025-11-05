"""Heuristics processing stage - extract and filter sentences using rule-based methods."""

import os
import re
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets

from ..config import LOG_DIR
from ..text_processing import clean_wiki_markup, split_into_sentences, looks_like_sentence_with_reason, soft_clean
from ..utils.io_utils import ensure_output_dir, get_text_column, get_title_column
from ..utils.logging import ensure_logdir, reset_log, write_jsonl
from .base import stage_path


def extract_sentences_with_logs(batch: Dict[str, List[Any]], split_name: str) -> Dict[str, List[Any]]:
    """Extract and filter sentences, outputting ONE RECORD PER SENTENCE.
    
    Args:
        batch: Batch of examples from the dataset
        split_name: Name of the dataset split being processed
        
    Returns:
        Dictionary with extracted sentence data
    """
    out = {
        "title": [],
        "source_idx": [],
        "sentence_idx": [],
        "split": [],
        "sentence": [],
        "source_num_sentences_total": [],
    }
    heur_log_batch = []

    for i in range(len(next(iter(batch.values())))):
        ex = {k: batch[k][i] for k in batch}
        text_col = get_text_column(ex)
        title_col = get_title_column(ex)

        raw = ex.get(text_col, "") if text_col else ""
        title = ex.get(title_col, "") if title_col else ""
        idx = ex.get("__idx__", -1)

        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""

        cleaned_doc = clean_wiki_markup(raw)
        fragments = [ln.strip() for ln in re.split(r"[\r\n]+", cleaned_doc)]

        accepted_sentences: List[str] = []
        total_candidates = 0

        # Import patterns for fragment filtering
        from ..text_processing.heuristics import get_heading_pattern, get_list_pattern, get_table_pattern
        heading_re = get_heading_pattern()
        list_re = get_list_pattern()
        table_re = get_table_pattern()

        for frag in fragments:
            if not frag:
                continue
            if heading_re.match(frag) or list_re.match(frag) or table_re.match(frag):
                # Log outright fragment drop
                heur_log_batch.append({
                    "stage": "heuristics",
                    "split": split_name,
                    "source_idx": idx,
                    "title": title,
                    "fragment_before": frag,
                    "sentence_after": "",
                    "accepted": False,
                    "filter_reason": "block_fragment",
                })
                continue

            candidates = split_into_sentences(frag)
            if not candidates:
                heur_log_batch.append({
                    "stage": "heuristics",
                    "split": split_name,
                    "source_idx": idx,
                    "title": title,
                    "fragment_before": frag,
                    "sentence_after": "",
                    "accepted": False,
                    "filter_reason": "no_sentences",
                })
                continue

            for cand in candidates:
                total_candidates += 1
                ok, reason = looks_like_sentence_with_reason(cand)
                if ok:
                    final = soft_clean(cand)
                    accepted_sentences.append(final)
                    heur_log_batch.append({
                        "stage": "heuristics",
                        "split": split_name,
                        "source_idx": idx,
                        "title": title,
                        "fragment_before": frag,
                        "sentence_after": final,
                        "accepted": True,
                        "filter_reason": None,
                    })
                else:
                    heur_log_batch.append({
                        "stage": "heuristics",
                        "split": split_name,
                        "source_idx": idx,
                        "title": title,
                        "fragment_before": frag,
                        "sentence_after": cand,
                        "accepted": False,
                        "filter_reason": reason,
                    })

        # Output ONE RECORD PER ACCEPTED SENTENCE
        for sent_idx, sentence in enumerate(accepted_sentences):
            out["title"].append(title)
            out["source_idx"].append(idx)
            out["sentence_idx"].append(sent_idx)
            out["split"].append(split_name)
            out["sentence"].append(sentence)
            out["source_num_sentences_total"].append(total_candidates)

    # Flush logs for this map batch
    # Note: Multiprocessing (num_proc > 1) causes race conditions with log writes.
    # Keep num_proc=1 for reliable logging, or accept that logs may be incomplete.
    if heur_log_batch:
        write_jsonl(os.path.join(LOG_DIR, "01_heuristics.jsonl"), heur_log_batch)
    return out


def run_stage_heuristics(args) -> str:
    """Run the heuristics processing stage.
    
    Args:
        args: Argument namespace with processing configuration
        
    Returns:
        Path to the output file
    """
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)  # Ensure logs directory exists
    reset_log("01_heuristics.jsonl")

    print(f"[heuristics] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    parts = []
    for split_name, split in ds.items():
        with_idx = split.map(lambda ex, idx: {"__idx__": idx}, with_indices=True)
        if getattr(args, 'test_limit', None):
            limit = min(args.test_limit, len(with_idx))
            with_idx = with_idx.select(range(limit))
            print(f"[heuristics][TEST] Using {len(with_idx)} of {len(split)} rows from split '{split_name}'.")

        sents = with_idx.map(
            lambda batch: extract_sentences_with_logs(batch, split_name),
            batched=True,
            remove_columns=with_idx.column_names,
            num_proc=args.num_proc
        )
        parts.append(sents)

    all_records: Dataset = concatenate_datasets(parts)
    print(f"[heuristics] Processed {len(all_records):,} sentences (1 per record).")

    df = all_records.to_pandas()
    df["row_id"] = df.index.astype(int)
    df["sentence"] = df["sentence"].fillna("").astype(str)
    df["source_idx"] = df["source_idx"].fillna(-1).astype(int)
    df["sentence_idx"] = df["sentence_idx"].fillna(0).astype(int)
    df["source_num_sentences_total"] = df["source_num_sentences_total"].fillna(0).astype(int)
    
    # Show stats
    num_sources = df["source_idx"].nunique()
    avg_sentences_per_source = len(df) / num_sources if num_sources > 0 else 0
    print(f"[heuristics] From {num_sources:,} source documents, avg {avg_sentences_per_source:.1f} sentences/doc")

    out_path = stage_path(args, "heuristics")
    df.to_parquet(out_path, index=False)
    print(f"[heuristics] Saved {out_path}")
    return out_path
