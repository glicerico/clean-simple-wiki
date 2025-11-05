"""LLM processing stage - use language models for sentence refinement."""

import os
import time
from typing import Optional, Set, List, Dict, Any

import pandas as pd

from ..config import LOG_DIR, DROP_THRESHOLD, KEEP_THRESHOLD, LLM_MAX_RETRIES, DEFAULT_LLM_MODE
from ..models.llm_processor import LLMProcessor, create_openai_client
from ..batch_processing import BatchManager, group_sentences_by_context, apply_llm_result_group
from ..utils.io_utils import ensure_output_dir
from ..utils.logging import ensure_logdir, reset_log, write_jsonl, read_jsonl, append_jsonl
from .base import stage_path


def run_stage_llm(args, input_path: Optional[str] = None) -> str:
    """Run the LLM processing stage.
    
    Args:
        args: Argument namespace with processing configuration
        input_path: Optional path to input file (defaults to classifier output)
        
    Returns:
        Path to the output file
    """
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)

    llm_mode = getattr(args, "llm_mode", DEFAULT_LLM_MODE)
    allowed_modes = {"online", "batch_prepare", "batch_submit", "batch_collect"}
    if llm_mode not in allowed_modes:
        raise ValueError(f"Unsupported --llm_mode '{llm_mode}'. Allowed values: {sorted(allowed_modes)}")

    stage_output = stage_path(args, "llm")

    # Handle batch submission mode
    if llm_mode == "batch_submit":
        batch_manager = BatchManager(args)
        run_dir = batch_manager.submit_batch_requests()
        print(f"[batch] Submit complete for run '{args.batch_run_name}' ({run_dir})")
        return stage_output

    # Set up checkpoint handling for online mode
    checkpoint_path = os.path.join(LOG_DIR, "llm_checkpoint.jsonl")
    completed_ids: Set[int] = set()
    if llm_mode == "online":
        if not getattr(args, 'no_resume', False) and os.path.exists(checkpoint_path):
            print(f"[llm] Loading checkpoint from {checkpoint_path}")
            checkpoint_records = read_jsonl(checkpoint_path)
            completed_ids = {rec["row_id"] for rec in checkpoint_records if "row_id" in rec}
            print(f"[llm] Found {len(completed_ids)} already completed sentences")
        elif getattr(args, 'no_resume', False) and os.path.exists(checkpoint_path):
            print("[llm] --no-resume: Clearing checkpoint and starting fresh")
            os.remove(checkpoint_path)

    if llm_mode in {"online", "batch_collect"}:
        reset_log("03_llm.jsonl")

    # Find input file
    candidate_paths = [p for p in [input_path, stage_path(args, "classifier"), stage_path(args, "heuristics")] if p]
    source_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if source_path is None:
        raise FileNotFoundError("No classifier or heuristics output found. Run earlier stages first.")

    # Validate LLM provider
    llm_provider = getattr(args, 'llm_provider', 'openai')
    if llm_provider != "openai":
        raise NotImplementedError("Only OpenAI provider is implemented for LLM stage.")

    # Load and prepare data
    df = pd.read_parquet(source_path)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)

    if getattr(args, 'test_limit', None):
        original_len = len(df)
        limit = min(args.test_limit, len(df))
        df = df.head(limit).copy()
        print(f"[llm][TEST] Using {len(df)} of {original_len} sentences.")

    if "sentence" not in df.columns:
        raise ValueError("Input must have 'sentence' column.")

    df["sentence"] = df["sentence"].fillna("").astype(str)

    # Initialize missing columns
    for col, default in [("keep", True), ("confidence", None), ("decision_source", "heuristics"), ("score_classifier", None)]:
        if col not in df.columns:
            df[col] = default

    if "llm_pending" not in df.columns:
        df["llm_pending"] = False
    else:
        df["llm_pending"] = df["llm_pending"].fillna(False)

    df["row_id"] = df["row_id"].astype(int)

    # Find pending sentences
    pending_idxs = df.index[df["llm_pending"]].tolist()

    # Filter out completed sentences for online mode
    if llm_mode == "online" and completed_ids:
        original_pending = len(pending_idxs)
        pending_idxs = [idx for idx in pending_idxs if df.loc[idx, "row_id"] not in completed_ids]
        skipped = original_pending - len(pending_idxs)
        if skipped > 0:
            print(f"[llm] Skipping {skipped} sentences already in checkpoint")

    # Handle batch preparation mode
    if llm_mode == "batch_prepare":
        if not pending_idxs:
            raise RuntimeError("No sentences pending LLM review. Nothing to prepare for batch mode.")
        pending_subset = df.loc[pending_idxs, ["row_id", "sentence", "title", "split", "source_idx", "sentence_idx", "score_classifier"]].copy()
        batch_manager = BatchManager(args)
        model = getattr(args, 'openai_model', 'gpt-4o-mini')
        run_dir = batch_manager.prepare_batch_requests(model, pending_subset)
        print(f"[batch] Prepared batch run at {run_dir}")
        return stage_output

    # Handle case where no pending sentences exist
    if llm_mode == "online" and not pending_idxs:
        if completed_ids:
            print("[llm] All dubious sentences already processed (found in checkpoint).")
            print(f"[llm] Merging {len(completed_ids)} checkpoint results into dataframe...")
            checkpoint_records = read_jsonl(checkpoint_path)
            for rec in checkpoint_records:
                rid = rec["row_id"]
                matching_rows = df[df["row_id"] == rid]
                if len(matching_rows) > 0:
                    idx = matching_rows.index[0]
                    df.loc[idx, "sentence"] = rec.get("sentence_cleaned", rec.get("sentence_original", df.loc[idx, "sentence"]))
                    df.loc[idx, "keep"] = rec["keep"]
                    df.loc[idx, "confidence"] = rec["confidence"]
                    df.loc[idx, "decision_source"] = "llm"
                    df.loc[idx, "llm_pending"] = False
        else:
            print("[llm] No dubious sentences flagged for LLM review.")
            print("[llm] All sentences already decided by heuristics/classifier.")
        df.to_parquet(stage_output, index=False)
        return stage_output

    # Show processing statistics
    total_sentences = len(df)
    pct_dubious = (len(pending_idxs) / total_sentences * 100) if total_sentences > 0 else 0.0
    if llm_mode == "online":
        print(f"[llm] Found {len(pending_idxs):,} dubious sentences to process ({pct_dubious:.1f}% of total)")
        print(f"[llm] These scored between {DROP_THRESHOLD:.2f}-{KEEP_THRESHOLD:.2f} in classifier stage")
    else:
        print(f"[batch] Collecting results across {len(pending_idxs):,} rows currently marked llm_pending ({pct_dubious:.1f}% of total)")

    # Process sentences
    model = getattr(args, 'openai_model', 'gpt-4o-mini')
    llm_logs: List[Dict[str, Any]] = []
    row_id_to_index = {int(df.loc[idx, "row_id"]): idx for idx in df.index}

    if llm_mode == "batch_collect":
        batch_manager = BatchManager(args)
        processed_row_ids, run_dir = batch_manager.collect_batch_results(args, df, row_id_to_index, llm_logs, checkpoint_path)
        print(f"[batch] Batch collect merged {len(processed_row_ids):,} rows for run '{args.batch_run_name}'")
    else:
        # Online processing
        processor = LLMProcessor(model, enable_tracing=True)
        
        # Group sentences by context (same source document, nearby sentences)
        batch_size = getattr(args, 'llm_batch_size', 16)
        context_groups = group_sentences_by_context(df, pending_idxs, context_size=batch_size)
        total_groups = len(context_groups)
        
        print(f"[llm] Created {total_groups} context groups from {len(pending_idxs)} sentences")
        
        for group_index, group in enumerate(context_groups, start=1):
            print(f"[llm] Group {group_index}/{total_groups}: Processing {len(group['sentences'])} sentences from '{group['title']}'")
            if group['sentences']:
                print(f"[llm]   Sample: '{group['sentences'][0][:60]}...'")

            if group_index > 1:
                delay = 2.0
                print(f"[llm]   Rate limit delay: {delay}s...")
                time.sleep(delay)

            retries = 0
            batch_start = time.time()
            while True:
                try:
                    print("[llm]   Calling OpenAI API...")
                    result = processor.process_batch(group['sentences'])
                    elapsed = time.time() - batch_start
                    output_count = len(result.get('sentences', []))
                    print(f"[llm]   ✓ Response received in {elapsed:.1f}s, got {output_count} output sentences")
                    break
                except Exception as e:
                    retries += 1
                    print(f"[llm][WARN] API call failed (attempt {retries}/{LLM_MAX_RETRIES}): {e}")
                    if retries >= LLM_MAX_RETRIES:
                        print(f"[llm][ERROR] Group failed permanently after {LLM_MAX_RETRIES} retries: {e}")
                        print("[llm][ERROR] Keeping sentences as llm_pending for retry on next run")
                        result = {"sentences": [], "confidence": 0.0}
                        batch_error_path = os.path.join(LOG_DIR, f"llm_group_error_{group_index}_{int(time.time())}.jsonl")
                        batch_error_records = []
                        from ..models.llm_processor import format_llm_payload
                        payload_size = len(format_llm_payload(group['sentences']))
                        for i, row_id in enumerate(group['row_ids']):
                            df_idx = row_id_to_index.get(row_id)
                            error_record = {
                                "timestamp": time.time(),
                                "group_num": group_index,
                                "error_type": "api_timeout" if "timeout" in str(e).lower() else "api_error",
                                "error_detail": str(e),
                                "retry_count": retries,
                                "row_id": int(row_id),
                                "split": str(group['split']),
                                "source_idx": int(group['source_idx']),
                                "title": str(group['title']),
                                "sentence": group['sentences'][i],
                                "sentence_length": len(group['sentences'][i]),
                                "score_classifier": float(df.loc[df_idx, "score_classifier"]) if (df_idx is not None and "score_classifier" in df.columns and df.loc[df_idx, "score_classifier"] is not None) else None,
                                "group_size": len(group['sentences']),
                                "payload_size": payload_size,
                            }
                            batch_error_records.append(error_record)
                        append_jsonl(batch_error_path, batch_error_records)
                        print(f"[llm][ERROR] Detailed group error saved to {batch_error_path}")
                        break
                    wait_time = 1.5 * retries
                    print(f"[llm]   Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            checkpoint_batch = []
            processed_count = apply_llm_result_group(df, group, result, llm_logs, checkpoint_batch)
            
            if checkpoint_batch:
                append_jsonl(checkpoint_path, checkpoint_batch)
                print(f"[llm]   ✓ Processed {len(group['sentences'])} → {processed_count} sentences, saved to checkpoint")

    # Write logs
    write_jsonl(os.path.join(LOG_DIR, "03_llm.jsonl"), llm_logs)

    # Check for remaining pending sentences
    print(f"[llm] DEBUG: Checking llm_pending status in dataframe...")
    print(f"[llm] DEBUG: Total rows in df: {len(df)}")
    print(f"[llm] DEBUG: llm_pending column type: {df['llm_pending'].dtype}")
    print(f"[llm] DEBUG: llm_pending value counts:\n{df['llm_pending'].value_counts()}")

    still_pending_mask = df["llm_pending"] == True
    num_still_pending = still_pending_mask.sum()
    print(f"[llm] DEBUG: num_still_pending = {num_still_pending}")

    if num_still_pending > 0:
        print(f"[llm] ⚠️ {num_still_pending} sentences still pending after this run")
        still_pending_ids = df.loc[still_pending_mask, "row_id"].tolist()
        print(f"[llm] Pending row_ids: {still_pending_ids[:10]}{'...' if len(still_pending_ids) > 10 else ''}")

        failed_batches_path = os.path.join(LOG_DIR, "llm_failed_batches.jsonl")
        failed_batch_records = []
        for idx in df.loc[still_pending_mask].index:
            row = df.loc[idx]
            failed_batch_records.append({
                "timestamp": time.time(),
                "row_id": int(row["row_id"]),
                "split": str(row["split"]),
                "source_idx": int(row["source_idx"]),
                "sentence_idx": int(row.get("sentence_idx", 0)),
                "title": str(row["title"]),
                "sentence": str(row["sentence"]),
                "score_classifier": float(row.get("score_classifier", 0.0)),
                "reason": "batch_processing_failed"
            })

        if failed_batch_records:
            append_jsonl(failed_batches_path, failed_batch_records)
            print(f"[llm] Dumped {len(failed_batch_records)} failed sentences to {failed_batches_path}")

    # Merge checkpoint results
    if os.path.exists(checkpoint_path):
        print(f"[llm] Merging all checkpoint results into final dataframe...")
        all_checkpoint_records = read_jsonl(checkpoint_path)
        merged_count = 0
        for rec in all_checkpoint_records:
            rid = rec["row_id"]
            matching_rows = df[df["row_id"] == rid]
            if len(matching_rows) > 0:
                idx = matching_rows.index[0]
                if df.loc[idx, "llm_pending"]:
                    df.loc[idx, "sentence"] = rec.get("sentence_cleaned", rec.get("sentence_original", df.loc[idx, "sentence"]))
                    df.loc[idx, "keep"] = rec["keep"]
                    df.loc[idx, "confidence"] = rec["confidence"]
                    df.loc[idx, "decision_source"] = "llm"
                    df.loc[idx, "llm_pending"] = False
                    merged_count += 1

        if merged_count > 0:
            print(f"[llm] Merged {merged_count} additional results from checkpoint")

        if num_still_pending == 0:
            print(f"[llm] All sentences processed successfully. Removing checkpoint file.")
            os.remove(checkpoint_path)
        else:
            print(f"[llm] ⚠️ {num_still_pending} sentences still pending (failed batches).")
            print(f"[llm] Checkpoint preserved at {checkpoint_path}")
            print(f"[llm] Re-run to retry failed batches, or use --no-resume to start fresh.")
            print(f"[llm] Failed batch details saved to logs/llm_failed_batches.jsonl")

    df.to_parquet(stage_output, index=False)
    print(f"[llm] Saved {stage_output}")
    return stage_output
