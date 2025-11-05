"""Utility functions for batch processing operations."""

import json
from typing import List, Dict, Any, Tuple, Set

import pandas as pd

from ..models.llm_processor import parse_llm_response_text


def group_sentences_by_context(df: pd.DataFrame, pending_idxs: List[int], context_size: int = 5) -> List[Dict[str, Any]]:
    """Group sentences by source document and proximity for context-aware processing.
    
    Args:
        df: DataFrame containing sentences
        pending_idxs: List of indices that need processing
        context_size: Maximum number of sentences per group
        
    Returns:
        List of group dictionaries with sentence information
    """
    groups = []
    
    # Group by source_idx first
    source_groups = {}
    for idx in pending_idxs:
        source_idx = df.loc[idx, "source_idx"]
        if source_idx not in source_groups:
            source_groups[source_idx] = []
        source_groups[source_idx].append(idx)
    
    # Within each source, create context groups
    for source_idx, indices in source_groups.items():
        # Sort by sentence_idx to maintain order
        indices.sort(key=lambda idx: df.loc[idx, "sentence_idx"])
        
        # Create groups of up to context_size sentences
        for i in range(0, len(indices), context_size):
            group_indices = indices[i:i + context_size]
            if group_indices:  # Only create non-empty groups
                group_info = {
                    "indices": group_indices,
                    "source_idx": source_idx,
                    "title": df.loc[group_indices[0], "title"],
                    "split": df.loc[group_indices[0], "split"],
                    "sentences": [df.loc[idx, "sentence"] for idx in group_indices],
                    "row_ids": [int(df.loc[idx, "row_id"]) for idx in group_indices]
                }
                groups.append(group_info)
    
    return groups


def apply_llm_result_group(
    df: pd.DataFrame,
    group: Dict[str, Any],
    result: Dict[str, Any],
    llm_logs: List[Dict[str, Any]],
    checkpoint_batch: List[Dict[str, Any]],
) -> int:
    """Apply LLM results to a group of sentences, handling variable output.
    
    Args:
        df: DataFrame to update
        group: Group information dictionary
        result: LLM processing result
        llm_logs: List to append log entries to
        checkpoint_batch: List to append checkpoint entries to
        
    Returns:
        Number of output sentences produced
    """
    input_indices = group["indices"]
    input_sentences = group["sentences"]
    output_sentences = result.get("sentences", [])
    confidence = float(result.get("confidence", 0.0))
    
    # Mark all original sentences as processed
    for idx in input_indices:
        df.loc[idx, "llm_pending"] = False
        df.loc[idx, "decision_source"] = "llm"
        df.loc[idx, "confidence"] = confidence
    
    # Handle different output scenarios
    if not output_sentences:
        # No output sentences - mark all as dropped
        for idx in input_indices:
            df.loc[idx, "keep"] = False
            
            log_entry = {
                "stage": "llm",
                "row_id": int(df.loc[idx, "row_id"]),
                "split": str(df.loc[idx, "split"]),
                "source_idx": int(df.loc[idx, "source_idx"]),
                "title": str(df.loc[idx, "title"]),
                "sentence_original": df.loc[idx, "sentence"],
                "sentence_cleaned": "",
                "keep": False,
                "confidence": confidence,
                "transformation": "dropped"
            }
            llm_logs.append(log_entry)
            checkpoint_batch.append(log_entry)
        return 0
    
    elif len(output_sentences) == len(input_sentences):
        # 1:1 mapping - replace each sentence
        for idx, new_sentence in zip(input_indices, output_sentences):
            df.loc[idx, "sentence"] = new_sentence
            df.loc[idx, "keep"] = True
            
            log_entry = {
                "stage": "llm",
                "row_id": int(df.loc[idx, "row_id"]),
                "split": str(df.loc[idx, "split"]),
                "source_idx": int(df.loc[idx, "source_idx"]),
                "title": str(df.loc[idx, "title"]),
                "sentence_original": input_sentences[input_indices.index(idx)],
                "sentence_cleaned": new_sentence,
                "keep": True,
                "confidence": confidence,
                "transformation": "1to1"
            }
            llm_logs.append(log_entry)
            checkpoint_batch.append(log_entry)
        return len(output_sentences)
    
    elif len(output_sentences) < len(input_sentences):
        # Consolidation - use first N sentences, drop the rest
        for i, idx in enumerate(input_indices):
            if i < len(output_sentences):
                df.loc[idx, "sentence"] = output_sentences[i]
                df.loc[idx, "keep"] = True
                transformation = "consolidated"
            else:
                df.loc[idx, "keep"] = False
                transformation = "dropped_in_consolidation"
            
            log_entry = {
                "stage": "llm",
                "row_id": int(df.loc[idx, "row_id"]),
                "split": str(df.loc[idx, "split"]),
                "source_idx": int(df.loc[idx, "source_idx"]),
                "title": str(df.loc[idx, "title"]),
                "sentence_original": input_sentences[i],
                "sentence_cleaned": output_sentences[i] if i < len(output_sentences) else "",
                "keep": i < len(output_sentences),
                "confidence": confidence,
                "transformation": transformation
            }
            llm_logs.append(log_entry)
            checkpoint_batch.append(log_entry)
        return len(output_sentences)
    
    else:
        # Expansion - need to create new rows
        # Use first sentence slot for first output, create new rows for the rest
        for i, new_sentence in enumerate(output_sentences):
            if i < len(input_indices):
                # Use existing row
                idx = input_indices[i]
                df.loc[idx, "sentence"] = new_sentence
                df.loc[idx, "keep"] = True
                original_sentence = input_sentences[i]
            else:
                # Create new row by duplicating the last input row
                base_idx = input_indices[-1]
                new_idx = len(df)
                df.loc[new_idx] = df.loc[base_idx].copy()
                df.loc[new_idx, "row_id"] = new_idx
                df.loc[new_idx, "sentence"] = new_sentence
                df.loc[new_idx, "sentence_idx"] = df.loc[base_idx, "sentence_idx"] + (i - len(input_indices) + 1)
                df.loc[new_idx, "keep"] = True
                df.loc[new_idx, "llm_pending"] = False
                df.loc[new_idx, "decision_source"] = "llm"
                df.loc[new_idx, "confidence"] = confidence
                original_sentence = f"[EXPANDED FROM: {input_sentences[-1][:50]}...]"
            
            log_entry = {
                "stage": "llm",
                "row_id": int(df.loc[new_idx if i >= len(input_indices) else input_indices[i], "row_id"]),
                "split": str(group["split"]),
                "source_idx": int(group["source_idx"]),
                "title": str(group["title"]),
                "sentence_original": original_sentence,
                "sentence_cleaned": new_sentence,
                "keep": True,
                "confidence": confidence,
                "transformation": "expanded" if i >= len(input_indices) else "split"
            }
            llm_logs.append(log_entry)
            checkpoint_batch.append(log_entry)
        
        return len(output_sentences)


def parse_batch_output_file(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse OpenAI batch output file into results and errors.
    
    Args:
        path: Path to the batch output file
        
    Returns:
        Tuple of (results, errors) lists
    """
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"custom_id": None, "error": f"json_decode_error: {exc}", "line": line[:200]})
                continue

            custom_id = rec.get("custom_id")
            response_obj = rec.get("response")
            if response_obj and response_obj.get("status_code", 0) == 200:
                body = response_obj.get("body")
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except json.JSONDecodeError:
                        errors.append({"custom_id": custom_id, "error": "body_decode_error", "body": body[:200]})
                        continue
                if not isinstance(body, dict):
                    errors.append({"custom_id": custom_id, "error": "unexpected_body_type"})
                    continue
                choices = body.get("choices", [])
                if not choices:
                    errors.append({"custom_id": custom_id, "error": "no_choices", "body": body})
                    continue
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    text = "".join(text_parts)
                else:
                    errors.append({"custom_id": custom_id, "error": "unsupported_content_type"})
                    continue

                try:
                    parsed_items = parse_llm_response_text(text)
                except Exception as exc:
                    errors.append({"custom_id": custom_id, "error": f"parse_error: {exc}", "response_text": text[:2000]})
                    continue
                for item in parsed_items:
                    item["custom_id"] = custom_id
                results.extend(parsed_items)
            else:
                errors.append({
                    "custom_id": custom_id,
                    "error": rec.get("error") or response_obj,
                })

    return results, errors
