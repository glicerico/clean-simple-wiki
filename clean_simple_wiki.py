#!/usr/bin/env python3
"""
Clean Simple Wikipedia with auditable, staged processing.

Stages (re-runnable):
  heuristics → classifier → llm → finalize

Logs (default ./logs):
  01_heuristics.jsonl      # extraction + heuristic filtering (before/after + reason)
  02_classifier.jsonl      # classifier scores + triage bucket
  03_llm.jsonl             # LLM decisions (before/after + keep + confidence)
  04_final_decisions.jsonl # final record for each kept/dropped item

Final artifacts (default ./):
  simple_wiki_clean.jsonl
  simple_wiki_clean.parquet
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Any, Iterable, Tuple, Optional, Set

import orjson
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Optional deps
try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai
    _LANGSMITH_AVAILABLE = True
except Exception:
    _LANGSMITH_AVAILABLE = False
    # Create dummy decorator if langsmith not available
    def traceable(func):
        return func
    wrap_openai = None

# ---------------- Config ----------------
DEFAULT_DATASET = "rahular/simple-wikipedia"
OUT_PREFIX = "simple_wiki_clean"
DEFAULT_JSONL = f"{OUT_PREFIX}.jsonl"
DEFAULT_PARQUET = f"{OUT_PREFIX}.parquet"
LOG_DIR = "logs"

DEFAULT_NUM_PROC = 1  # Keep at 1 to ensure logs are written correctly
DEFAULT_BATCH_SIZE_CLASSIFIER = 64
MAX_SENT_CHARS = 1000

KEEP_THRESHOLD = 0.75
DROP_THRESHOLD = 0.35

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LLM_BATCH_SIZE = 16  # Conservative batch size for structured JSON output with coreference resolution
LLM_MAX_RETRIES = 1

DEFAULT_LLM_MODE = "online"
DEFAULT_BATCH_DIR_NAME = "batches"
DEFAULT_BATCH_REQUESTS_PER_FILE = 500
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"

DEFAULT_TEST_LIMIT = 100

LLM_TIMEOUT = 30  # Avoid hanging when no response from LLM

# ---------------- Heuristics ----------------
_url_re = re.compile(r"https?://\S+|www\.\S+", re.I)
_ref_re = re.compile(r"\[\d+\]")
_cite_re = re.compile(r"\{\{.*?\}\}")
_file_re = re.compile(r"\[\[(File|Image):.*?\]\]", re.I)
_link_re = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_heading_re = re.compile(r"^\s*={2,}.*?={2,}\s*$")
_list_re = re.compile(r"^\s*([*#:-]|\d+\.)\s+")
_table_re = re.compile(r"^\s*\{\|.*|\|\}|\|\-|\!|\|\s", re.I)
_cleanup_spaces = re.compile(r"\s+")

def clean_wiki_markup(text: str) -> str:
    text = _url_re.sub("", text)
    text = _cite_re.sub(" ", text)
    text = _file_re.sub(" ", text)
    text = _link_re.sub(r"\1", text)
    text = _ref_re.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = _cleanup_spaces.sub(" ", text).strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    """Use NLTK's robust sentence tokenizer."""
    from nltk.tokenize import sent_tokenize
    text = text.strip()
    if not text:
        return []
    return sent_tokenize(text)

def looks_like_sentence_with_reason(s: str) -> Tuple[bool, Optional[str]]:
    if len(s) < 15:
        return False, "length"
    if len(s) > MAX_SENT_CHARS:
        return False, "length"
    if _heading_re.match(s):
        return False, "heading"
    if _list_re.match(s):
        return False, "list"
    if _table_re.match(s):
        return False, "table"
    if not re.search(r"[A-Za-z]", s):
        return False, "no_letters"
    if len(s.split()) < 3:
        return False, "too_few_words"
    # Require sentence-ending punctuation for short sentences
    if len(s.split()) < 8 and not s.rstrip().endswith(('.', '!', '?')):
        return False, "not_sentence_like"
    return True, None

def soft_clean(s: str) -> str:
    """Clean spacing but don't truncate - semantic parsers need complete sentences."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s([,;:])", r"\1", s)
    return s

# ---------------- Columns ----------------
TEXT_COL_CANDIDATES = ["text", "article", "content", "body"]
TITLE_COL_CANDIDATES = ["title", "heading"]

def pick_col(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in example and example[c] is not None:
            return c
    for k, v in example.items():
        if isinstance(v, str):
            return k
    return None

# ---------------- Logging utils ----------------
def ensure_logdir(path: Optional[str] = None):
    target = path or LOG_DIR
    if target:
        os.makedirs(target, exist_ok=True)

def write_jsonl(path: str, recs: Iterable[dict]):
    ensure_logdir(os.path.dirname(path) or ".")
    with open(path, "ab") as f:
        for r in recs:
            f.write(orjson.dumps(r))
            f.write(b"\n")

def reset_log(filename: str):
    log_dir = LOG_DIR
    if not log_dir:
        return None
    path = os.path.join(log_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    return path

def read_jsonl(path: str) -> List[dict]:
    """Read JSONL file and return list of dicts."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'rb') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def append_jsonl(path: str, records: List[dict]):
    """Append records to JSONL file (atomic writes)."""
    ensure_logdir(os.path.dirname(path) or ".")
    with open(path, 'ab') as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b'\n')
        f.flush()  # Ensure written to disk
        os.fsync(f.fileno())  # Force OS to write to disk

# ---------------- Extraction + Heuristics (with logging) ----------------
def extract_sentences_with_logs(batch: Dict[str, List[Any]], split_name: str) -> Dict[str, List[Any]]:
    """Extract and filter sentences, outputting ONE RECORD PER SENTENCE."""
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
        text_col = pick_col(ex, TEXT_COL_CANDIDATES)
        title_col = pick_col(ex, TITLE_COL_CANDIDATES) if TITLE_COL_CANDIDATES else None

        raw = ex.get(text_col, "")
        title = ex.get(title_col, "") if title_col else ""
        idx = ex.get("__idx__", -1)

        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""

        cleaned_doc = clean_wiki_markup(raw)
        fragments = [ln.strip() for ln in re.split(r"[\r\n]+", cleaned_doc)]

        accepted_sentences: List[str] = []
        total_candidates = 0

        for frag in fragments:
            if not frag:
                continue
            if _heading_re.match(frag) or _list_re.match(frag) or _table_re.match(frag):
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

# ---------------- Classifier ----------------
def load_tiny_classifier(model_name: str = "typeform/distilbert-base-uncased-mnli"):
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed.")
    return pipeline("zero-shot-classification", model=model_name, device=-1, truncation=True)

def classify_batch(clf, texts: List[str], batch_size: int) -> List[float]:
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
    keep, drop, gray = [], [], []
    for i, s in enumerate(scores):
        if s >= KEEP_THRESHOLD:
            keep.append(i)
        elif s <= DROP_THRESHOLD:
            drop.append(i)
        else:
            gray.append(i)
    return keep, drop, gray

# ---------------- LLM ----------------
LLM_SYSTEM_PROMPT = (
    "You are a language-to-knowledge conversion model.\n\n"
    "Your job is to transform text into clear, factual, standalone sentences that can be directly used for knowledge extraction and database ingestion.\n\n"
    "Detailed Guidelines\n\n"
    "    Input type:\n\n"
    "    The input text comes from Simple English Wikipedia and may include sentences, fragments, or explanatory phrasing.\n\n"
    "    Objective:\n\n"
    "    Convert the text into a list of fully independent factual sentences.\n\n"
    "    Each sentence must:\n\n"
    "        Convey one complete, atomic fact.\n\n"
    "        Be understandable without any previous context.\n\n"
    "        Include explicit subjects and objects (no pronouns or implicit references).\n\n"
    "        Avoid transitional or connective words (e.g., \"however,\" \"therefore,\" \"this,\" \"these,\" \"such,\" \"as mentioned,\" etc.).\n\n"
    "        Preserve the factual accuracy of the original content.\n\n"
    "    Filtering Rule:\n\n"
    "        Remove any line or phrase that does not express a factual statement suitable for knowledge extraction.\n\n"
    "        Examples to remove include: definitions that merely restate words (\"This means that…\"), stylistic commentary, or redundant explanations.\n\n"
    "    Language style:\n\n"
    "        Keep the tone factual and formal.\n\n"
    "        Do not simplify scientific or technical terms further — use precise terminology.\n\n"
    "        Do not add new information or interpretations.\n\n"
    "    Output format:\n\n"
    "        Output one factual, standalone sentence per line.\n\n"
    "        Do not number or bullet the lines.\n\n"
    "        Maintain consistent capitalization and punctuation."
)
LLM_USER_TEMPLATE = """TASK:
Return a JSON array with objects:
{{"sentences": ["sentence1", "sentence2", ...], "confidence": <0..1>}}

EXAMPLE INPUT:

Helium is a chemical element. It usually has two neutrons, but some helium atoms have only one. These atoms are still helium because the number of protons defines the element. However, they are not normal helium either.

EXAMPLE OUTPUT:

{{"sentences": ["Helium is a chemical element.", "A typical helium atom contains two neutrons.", "Some helium atoms contain one neutron.", "An atom that contains two protons is defined as helium."], "confidence": 0.9}}

INPUT TEXT:
{payload}

Transform the above text into clear, factual, standalone sentences following the guidelines.
"""

def format_llm_payload(chunks: List[str]) -> str:
    """Format payload as a block of sentences for context-aware processing."""
    return "\n\n".join(chunks)


def build_llm_messages(chunks: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (user_payload, messages) for the chat completion call."""
    content = LLM_USER_TEMPLATE.format(payload=format_llm_payload(chunks))
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return content, messages


def parse_llm_response_text(text: str) -> Dict[str, Any]:
    """Parse the JSON response returned by the LLM into structured record."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode LLM response as JSON: {exc}\n{text[:2000]}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from LLM response, got {type(obj)}")
    
    if "sentences" not in obj:
        raise ValueError(f"LLM response missing 'sentences' field: {obj}")
    
    sentences = obj.get("sentences", [])
    if not isinstance(sentences, list):
        raise ValueError(f"Expected 'sentences' to be a list, got {type(sentences)}")
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        if isinstance(sentence, str) and sentence.strip():
            cleaned_sentences.append(sentence.strip())
    
    return {
        "sentences": cleaned_sentences,
        "confidence": float(obj.get("confidence", 0.0)),
    }

@traceable(name="llm_batch_review")
def call_openai_json(client, model: str, chunks: List[str]) -> Dict[str, Any]:
    content, messages = build_llm_messages(chunks)
    
    # Debug: show payload size
    payload_size = len(content)
    num_lines = content.count('\n')
    print(f"[llm]     Payload: {payload_size} chars, {num_lines} lines, {len(chunks)} input sentences")
    
    # Debug: show first few lines of payload
    first_lines = '\n'.join(content.split('\n')[:3])
    print(f"[llm]     First lines:\n{first_lines}")
    
    # Debug: show input sentences
    if len(chunks) <= 5:
        print(f"[llm]     All input sentences:")
        for i, sentence in enumerate(chunks):
            print(f"[llm]       {i+1}. {sentence[:100]}...")
    else:
        print(f"[llm]     First 3 input sentences:")
        for i, sentence in enumerate(chunks[:3]):
            print(f"[llm]       {i+1}. {sentence[:100]}...")
    
    print(f"[llm]     Sending to {model}... ")
    api_start = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
        response_format={"type":"json_object"},
        timeout=LLM_TIMEOUT
    )
    api_elapsed = time.time() - api_start
    print(f"[llm]     API returned in {api_elapsed:.1f}s")
    
    text = resp.choices[0].message.content
    print(f"[llm]     Response length: {len(text)} chars")
    print(f"[llm]     Parsing JSON...")
    
    parsed = parse_llm_response_text(text)
    print(f"[llm]     Parsed {len(parsed.get('sentences', []))} output sentences from JSON")
    return parsed


def create_openai_client(enable_tracing: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    if enable_tracing and _LANGSMITH_AVAILABLE and wrap_openai:
        client = wrap_openai(client)
        print("[llm] LangSmith tracing enabled")
    return client


BATCH_MANIFEST_FILENAME = "manifest.json"
BATCH_PENDING_PARQUET = "pending_sentences.parquet"
BATCH_RESPONSES_SUBDIR = "responses"
BATCH_REQUESTS_SUBDIR = "requests"
BATCH_META_SUBDIR = "meta"


def _timestamp_slug() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def resolve_batch_base_dir(args) -> str:
    base_dir = args.batch_dir or os.path.join(LOG_DIR, DEFAULT_BATCH_DIR_NAME)
    ensure_output_dir(base_dir)
    return os.path.abspath(base_dir)


def resolve_batch_run_dir(args, base_dir: str, create_if_missing: bool = False) -> Tuple[str, str]:
    run_name = args.batch_run_name
    if not run_name:
        if create_if_missing:
            run_name = _timestamp_slug()
            args.batch_run_name = run_name
        else:
            run_name = find_latest_batch_run(base_dir)
            if not run_name:
                raise RuntimeError("No batch runs found. Specify --batch_run_name or run with --llm_mode batch_prepare first.")
            args.batch_run_name = run_name
    run_dir = os.path.join(base_dir, run_name)
    if create_if_missing:
        ensure_output_dir(run_dir)
        ensure_output_dir(os.path.join(run_dir, BATCH_REQUESTS_SUBDIR))
        ensure_output_dir(os.path.join(run_dir, BATCH_RESPONSES_SUBDIR))
        ensure_output_dir(os.path.join(run_dir, BATCH_META_SUBDIR))
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Batch run directory not found: {run_dir}")
    return run_name, run_dir


def batch_manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, BATCH_META_SUBDIR, BATCH_MANIFEST_FILENAME)


def load_batch_manifest(run_dir: str) -> Dict[str, Any]:
    path = batch_manifest_path(run_dir)
    if not os.path.exists(path):
        raise RuntimeError(f"Batch manifest not found at {path}. Run batch_prepare first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_batch_manifest(run_dir: str, manifest: Dict[str, Any]):
    path = batch_manifest_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def find_latest_batch_run(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            candidates.append((os.path.getmtime(path), name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_batch_output_file(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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


def group_sentences_by_context(df: pd.DataFrame, pending_idxs: List[int], context_size: int = 5) -> List[Dict[str, Any]]:
    """Group sentences by source document and proximity for context-aware processing."""
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
    """Apply LLM results to a group of sentences, handling variable output."""
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


def prepare_llm_batch_requests(args, model: str, pending_df: pd.DataFrame) -> str:
    if pending_df.empty:
        raise RuntimeError("No sentences pending LLM review. Nothing to prepare for batch mode.")

    base_dir = resolve_batch_base_dir(args)
    run_name, run_dir = resolve_batch_run_dir(args, base_dir, create_if_missing=True)
    requests_dir = os.path.join(run_dir, BATCH_REQUESTS_SUBDIR)
    responses_dir = os.path.join(run_dir, BATCH_RESPONSES_SUBDIR)
    meta_dir = os.path.join(run_dir, BATCH_META_SUBDIR)

    ensure_output_dir(requests_dir)
    ensure_output_dir(responses_dir)
    ensure_output_dir(meta_dir)

    pending_df = pending_df.copy()
    pending_df["row_id"] = pending_df["row_id"].astype(int)
    pending_df["sentence"] = pending_df["sentence"].fillna("").astype(str)
    pending_df["title"] = pending_df["title"].fillna("").astype(str)

    pending_parquet_path = os.path.join(meta_dir, BATCH_PENDING_PARQUET)
    pending_df.to_parquet(pending_parquet_path, index=False)

    requests_per_file = max(1, getattr(args, "batch_requests_per_file", DEFAULT_BATCH_REQUESTS_PER_FILE))
    batch_size = max(1, args.llm_batch_size)

    row_ids = pending_df["row_id"].tolist()
    sentences = pending_df["sentence"].tolist()
    titles = pending_df["title"].tolist()

    manifest: Dict[str, Any] = {
        "run_name": run_name,
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "llm_batch_size": batch_size,
        "requests_per_file": requests_per_file,
        "total_sentences": len(row_ids),
        "total_requests": 0,
        "request_files": [],
        "pending_parquet": pending_parquet_path,
        "status": "prepared",
    }

    request_counter = 0
    file_index = 0
    current_fp = None
    current_info: Dict[str, Any] = {}

    def start_new_file():
        nonlocal file_index, current_fp, current_info
        if current_fp:
            current_fp.close()
            if current_info:
                manifest["request_files"].append(current_info)
        file_index += 1
        file_name = f"requests_{file_index:05d}.jsonl"
        file_path = os.path.join(requests_dir, file_name)
        current_fp = open(file_path, "w", encoding="utf-8")
        current_info = {
            "file_path": file_path,
            "num_requests": 0,
            "num_sentences": 0,
            "status": "prepared",
            "input_file_id": None,
            "batch_id": None,
            "submitted_at": None,
            "completed_at": None,
            "output_file_id": None,
            "error_file_id": None,
            "results_file": None,
            "results_merged": False,
            "error_message": None,
        }

    batches_generated = 0

    start_new_file()

    for start in range(0, len(row_ids), batch_size):
        row_id_batch = row_ids[start:start + batch_size]
        chunk_batch = sentences[start:start + batch_size]
        title_batch = titles[start:start + batch_size]
        request_counter += 1
        batches_generated += 1
        custom_id = f"{run_name}_req_{request_counter:07d}"
        _, messages = build_llm_messages(chunk_batch)
        request_record = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0.0,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
        }
        line = json.dumps(request_record, ensure_ascii=False)
        current_fp.write(line)
        current_fp.write("\n")

        current_info["num_requests"] += 1
        current_info["num_sentences"] += len(row_id_batch)

        if current_info["num_requests"] >= requests_per_file:
            start_new_file()

    # Close last file and append info
    if current_fp:
        current_fp.close()
        if current_info:
            manifest["request_files"].append(current_info)

    manifest["total_requests"] = batches_generated

    save_batch_manifest(run_dir, manifest)

    print(f"[batch] Prepared run '{run_name}' at {run_dir}")
    print(f"[batch] Pending sentences: {len(row_ids):,}")
    print(f"[batch] Requests generated: {batches_generated:,} across {len(manifest['request_files']):,} files")
    print(f"[batch] Example request file: {manifest['request_files'][0]['file_path'] if manifest['request_files'] else 'N/A'}")
    print(f"[batch] Pending sentences snapshot saved to {pending_parquet_path}")

    return run_dir


def submit_llm_batch_requests(args) -> str:
    base_dir = resolve_batch_base_dir(args)
    run_name, run_dir = resolve_batch_run_dir(args, base_dir, create_if_missing=False)
    manifest = load_batch_manifest(run_dir)
    client = create_openai_client(enable_tracing=False)
    completion_window = getattr(args, "batch_completion_window", DEFAULT_BATCH_COMPLETION_WINDOW)

    submitted = 0
    for entry in manifest.get("request_files", []):
        status = entry.get("status")
        if status not in {"prepared", "failed"}:
            continue
        file_path = entry["file_path"]
        if not os.path.exists(file_path):
            print(f"[batch][WARN] Request file missing, skipping: {file_path}")
            entry["status"] = "missing"
            entry["error_message"] = "request_file_missing"
            continue
        try:
            with open(file_path, "rb") as f:
                upload = client.files.create(file=f, purpose="batch")
            batch = client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata={"run_name": run_name, "source": "clean_simple_wiki"},
            )
            entry["status"] = batch.status or "submitted"
            entry["input_file_id"] = upload.id
            entry["batch_id"] = batch.id
            entry["submitted_at"] = time.time()
            entry["submitted_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            entry["error_message"] = None
            submitted += 1
            print(f"[batch] Submitted file {file_path} → batch_id={batch.id} status={entry['status']}")
        except Exception as exc:
            entry["status"] = "error"
            entry["error_message"] = str(exc)
            print(f"[batch][ERROR] Failed to submit {file_path}: {exc}")

    if submitted:
        manifest["status"] = "submitted"
        manifest["submitted_at"] = time.time()
        manifest["submitted_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    else:
        print("[batch] No request files submitted (all already submitted or missing)")

    save_batch_manifest(run_dir, manifest)
    return run_dir


def collect_llm_batch_results(
    args,
    df: pd.DataFrame,
    row_id_to_index: Dict[int, int],
    llm_logs: List[Dict[str, Any]],
    checkpoint_path: str,
) -> Tuple[Set[int], str]:
    base_dir = resolve_batch_base_dir(args)
    run_name, run_dir = resolve_batch_run_dir(args, base_dir, create_if_missing=False)
    manifest = load_batch_manifest(run_dir)
    client = create_openai_client(enable_tracing=False)
    responses_dir = os.path.join(run_dir, BATCH_RESPONSES_SUBDIR)
    ensure_output_dir(responses_dir)

    processed_row_ids: Set[int] = set()
    batch_errors: List[Dict[str, Any]] = []

    for entry in manifest.get("request_files", []):
        batch_id = entry.get("batch_id")
        if not batch_id:
            continue
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as exc:
            entry["error_message"] = str(exc)
            print(f"[batch][ERROR] Failed to retrieve batch {batch_id}: {exc}")
            continue

        entry["status"] = batch.status
        if getattr(batch, "output_file_id", None):
            entry["output_file_id"] = batch.output_file_id
        if getattr(batch, "error_file_id", None):
            entry["error_file_id"] = batch.error_file_id

        if batch.status == "completed":
            output_file_id = batch.output_file_id
            if not output_file_id:
                entry["error_message"] = "missing_output_file_id"
                continue

            output_path = entry.get("results_file") or os.path.join(responses_dir, f"{batch_id}_output.jsonl")
            if not os.path.exists(output_path):
                try:
                    content = client.files.content(output_file_id)
                    data = content.read() if hasattr(content, "read") else content
                    if isinstance(data, str):
                        data_bytes = data.encode("utf-8")
                    elif isinstance(data, bytes):
                        data_bytes = data
                    else:
                        data_bytes = json.dumps(data).encode("utf-8")
                    with open(output_path, "wb") as f:
                        f.write(data_bytes)
                    print(f"[batch] Downloaded results for batch {batch_id} → {output_path}")
                except Exception as exc:
                    entry["error_message"] = f"download_error: {exc}"
                    print(f"[batch][ERROR] Failed to download output for batch {batch_id}: {exc}")
                    continue

            entry["results_file"] = output_path

            parsed_results, parse_errors = parse_batch_output_file(output_path)
            checkpoint_batch: List[Dict[str, Any]] = []
            applied_count = 0
            for res in parsed_results:
                # Note: batch collection needs to be updated for new format
                # This is a placeholder - batch mode needs redesign
                processed_row_ids.add(int(res.get("row_id", 0)))
                applied_count += 1

            if checkpoint_batch:
                append_jsonl(checkpoint_path, checkpoint_batch)
            if parse_errors:
                batch_errors.extend(parse_errors)

            entry["results_merged"] = True
            entry["completed_at"] = time.time()
            entry["completed_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            entry["applied_count"] = applied_count
            entry["error_message"] = None
            print(f"[batch] Applied {applied_count} results from batch {batch_id}")

        elif batch.status in {"failed", "expired", "cancelled"}:
            entry["error_message"] = f"batch_status={batch.status}"
            print(f"[batch][WARN] Batch {batch_id} finished with status {batch.status}")

    if manifest.get("request_files") and all(entry.get("results_merged") for entry in manifest["request_files"]):
        manifest["status"] = "completed"
        manifest["completed_at"] = time.time()
        manifest["completed_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    save_batch_manifest(run_dir, manifest)

    if batch_errors:
        error_log_path = os.path.join(run_dir, BATCH_META_SUBDIR, "batch_errors.jsonl")
        append_jsonl(error_log_path, batch_errors)
        print(f"[batch][WARN] Logged {len(batch_errors)} batch response errors to {error_log_path}")

    return processed_row_ids, run_dir

# ---------------- Utilities ----------------
def batched(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# ---------------- Stage Helpers ----------------
def stage_path(args, stage_name: str) -> str:
    filename = f"{args.out_prefix}_{stage_name}.parquet"
    return os.path.join(args.output_dir, filename)

def analyze_failed_batches(log_dir: str = LOG_DIR):
    """Analyze failed batch logs to help debug LLM processing issues."""
    failed_batches_path = os.path.join(log_dir, "llm_failed_batches.jsonl")
    
    if not os.path.exists(failed_batches_path):
        print(f"[analyze] No failed batches file found at {failed_batches_path}")
        return
    
    failed_records = read_jsonl(failed_batches_path)
    if not failed_records:
        print(f"[analyze] No failed batch records found")
        return
    
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

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run_stage_heuristics(args) -> str:
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)  # Ensure logs directory exists
    reset_log("01_heuristics.jsonl")

    print(f"[heuristics] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    parts = []
    for split_name, split in ds.items():
        with_idx = split.map(lambda ex, idx: {"__idx__": idx}, with_indices=True)
        if args.test_limit:
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

def run_stage_classifier(args, input_path: Optional[str] = None) -> str:
    """Classify individual sentences for quality."""
    if not args.use_classifier and input_path is None:
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

    if args.test_limit:
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

    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed (needed for classifier stage).")

    # Score all sentences
    clf = load_tiny_classifier(args.classifier_model)
    texts = df["sentence"].tolist()
    scores: List[float] = []
    
    print(f"[classifier] Scoring {len(texts):,} sentences...")
    for chunk in batched(texts, args.classifier_batch_size):
        scores.extend(classify_batch(clf, chunk, args.classifier_batch_size))

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

def run_stage_llm(args, input_path: Optional[str] = None) -> str:
    """Use LLM to review gray-zone sentences."""
    ensure_output_dir(args.output_dir)
    ensure_logdir(LOG_DIR)

    llm_mode = getattr(args, "llm_mode", DEFAULT_LLM_MODE)
    allowed_modes = {"online", "batch_prepare", "batch_submit", "batch_collect"}
    if llm_mode not in allowed_modes:
        raise ValueError(f"Unsupported --llm_mode '{llm_mode}'. Allowed values: {sorted(allowed_modes)}")

    stage_output = stage_path(args, "llm")

    if llm_mode == "batch_submit":
        run_dir = submit_llm_batch_requests(args)
        print(f"[batch] Submit complete for run '{args.batch_run_name}' ({run_dir})")
        return stage_output

    checkpoint_path = os.path.join(LOG_DIR, "llm_checkpoint.jsonl")
    completed_ids: Set[int] = set()
    if llm_mode == "online":
        if not args.no_resume and os.path.exists(checkpoint_path):
            print(f"[llm] Loading checkpoint from {checkpoint_path}")
            checkpoint_records = read_jsonl(checkpoint_path)
            completed_ids = {rec["row_id"] for rec in checkpoint_records if "row_id" in rec}
            print(f"[llm] Found {len(completed_ids)} already completed sentences")
        elif args.no_resume and os.path.exists(checkpoint_path):
            print("[llm] --no-resume: Clearing checkpoint and starting fresh")
            os.remove(checkpoint_path)

    if llm_mode in {"online", "batch_collect"}:
        reset_log("03_llm.jsonl")

    candidate_paths = [p for p in [input_path, stage_path(args, "classifier"), stage_path(args, "heuristics")] if p]
    source_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if source_path is None:
        raise FileNotFoundError("No classifier or heuristics output found. Run earlier stages first.")

    if args.llm_provider != "openai":
        raise NotImplementedError("Only OpenAI provider is implemented for LLM stage.")
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed (needed for LLM stage).")

    df = pd.read_parquet(source_path)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)

    if args.test_limit:
        original_len = len(df)
        limit = min(args.test_limit, len(df))
        df = df.head(limit).copy()
        print(f"[llm][TEST] Using {len(df)} of {original_len} sentences.")

    if "sentence" not in df.columns:
        raise ValueError("Input must have 'sentence' column.")

    df["sentence"] = df["sentence"].fillna("").astype(str)

    for col, default in [("keep", True), ("confidence", None), ("decision_source", "heuristics"), ("score_classifier", None)]:
        if col not in df.columns:
            df[col] = default

    if "llm_pending" not in df.columns:
        df["llm_pending"] = False
    else:
        df["llm_pending"] = df["llm_pending"].fillna(False)

    df["row_id"] = df["row_id"].astype(int)

    pending_idxs = df.index[df["llm_pending"]].tolist()

    if llm_mode == "online" and completed_ids:
        original_pending = len(pending_idxs)
        pending_idxs = [idx for idx in pending_idxs if df.loc[idx, "row_id"] not in completed_ids]
        skipped = original_pending - len(pending_idxs)
        if skipped > 0:
            print(f"[llm] Skipping {skipped} sentences already in checkpoint")

    if llm_mode == "batch_prepare":
        if not pending_idxs:
            raise RuntimeError("No sentences pending LLM review. Nothing to prepare for batch mode.")
        pending_subset = df.loc[pending_idxs, ["row_id", "sentence", "title", "split", "source_idx", "sentence_idx", "score_classifier"]].copy()
        run_dir = prepare_llm_batch_requests(args, args.openai_model, pending_subset)
        print(f"[batch] Prepared batch run at {run_dir}")
        return stage_output

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

    total_sentences = len(df)
    pct_dubious = (len(pending_idxs) / total_sentences * 100) if total_sentences > 0 else 0.0
    if llm_mode == "online":
        print(f"[llm] Found {len(pending_idxs):,} dubious sentences to process ({pct_dubious:.1f}% of total)")
        print(f"[llm] These scored between {DROP_THRESHOLD:.2f}-{KEEP_THRESHOLD:.2f} in classifier stage")
    else:
        print(f"[batch] Collecting results across {len(pending_idxs):,} rows currently marked llm_pending ({pct_dubious:.1f}% of total)")

    model = args.openai_model
    llm_logs: List[Dict[str, Any]] = []
    row_id_to_index = {int(df.loc[idx, "row_id"]): idx for idx in df.index}

    if llm_mode == "batch_collect":
        processed_row_ids, run_dir = collect_llm_batch_results(args, df, row_id_to_index, llm_logs, checkpoint_path)
        print(f"[batch] Batch collect merged {len(processed_row_ids):,} rows for run '{args.batch_run_name}'")
    else:
        client = create_openai_client(enable_tracing=True)
        
        # Group sentences by context (same source document, nearby sentences)
        context_groups = group_sentences_by_context(df, pending_idxs, context_size=args.llm_batch_size)
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
                    result = call_openai_json(client, model, group['sentences'])
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

    write_jsonl(os.path.join(LOG_DIR, "03_llm.jsonl"), llm_logs)

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

def run_stage_finalize(args) -> str:
    """Finalize: output clean sentences (1 per record) ready for semantic parsing."""
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

    if args.test_limit:
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
    jsonl_path = args.output_jsonl or os.path.join(args.output_dir, f"{args.out_prefix}.jsonl")
    parquet_path = args.output_parquet or os.path.join(args.output_dir, f"{args.out_prefix}.parquet")
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

# ---------------- Main ----------------
def main():
    global LOG_DIR
    ap = argparse.ArgumentParser("Stageable Simple Wikipedia cleaner.")
    ap.add_argument("--stage", default="all", choices=["heuristics", "classifier", "llm", "finalize", "all", "analyze-failures"])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--out_prefix", default=OUT_PREFIX)
    ap.add_argument("--log_dir", default=LOG_DIR)
    ap.add_argument("--output_jsonl", default=None)
    ap.add_argument("--output_parquet", default=None)
    ap.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC, 
                    help="Number of processes for dataset mapping (default: 1 for reliable logging)")
    ap.add_argument("--use_classifier", action="store_true", help="When running --stage all, include classifier stage.")
    ap.add_argument("--classifier_model", default="typeform/distilbert-base-uncased-mnli")
    ap.add_argument("--classifier_batch_size", type=int, default=DEFAULT_BATCH_SIZE_CLASSIFIER)
    ap.add_argument("--use_llm", action="store_true", help="When running --stage all, include LLM stage.")
    ap.add_argument("--llm_provider", default=DEFAULT_LLM_PROVIDER, choices=["openai"])
    ap.add_argument("--openai_model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--llm_batch_size", type=int, default=LLM_BATCH_SIZE)
    ap.add_argument("--llm_mode", default=DEFAULT_LLM_MODE, choices=["online", "batch_prepare", "batch_submit", "batch_collect"], help="LLM execution mode (default: online).")
    ap.add_argument("--batch_dir", default=None, help="Directory for OpenAI batch payloads and results (default: <log_dir>/batches).")
    ap.add_argument("--batch_run_name", default=None, help="Identifier for OpenAI batch runs; default auto-generates a timestamp when preparing batches.")
    ap.add_argument("--batch_requests_per_file", type=int, default=DEFAULT_BATCH_REQUESTS_PER_FILE, help="Number of API requests to store per JSONL batch request file when preparing batches.")
    ap.add_argument("--batch_completion_window", default=DEFAULT_BATCH_COMPLETION_WINDOW, help="OpenAI batch completion window (e.g., '24h').")
    ap.add_argument("--test-limit", type=int, default=None, help="Limit processing to N records for testing (applies to all stages).")
    ap.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and reprocess all LLM sentences.")
    args = ap.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    if args.output_jsonl:
        args.output_jsonl = os.path.abspath(args.output_jsonl)
    if args.output_parquet:
        args.output_parquet = os.path.abspath(args.output_parquet)
    LOG_DIR = os.path.abspath(args.log_dir)
    if args.batch_dir:
        args.batch_dir = os.path.abspath(args.batch_dir)

    if args.stage == "heuristics":
        run_stage_heuristics(args)
    elif args.stage == "classifier":
        if not args.use_classifier:
            args.use_classifier = True
        run_stage_classifier(args)
    elif args.stage == "llm":
        if not args.use_llm:
            args.use_llm = True
        run_stage_llm(args)
    elif args.stage == "finalize":
        run_stage_finalize(args)
    elif args.stage == "analyze-failures":
        analyze_failed_batches(LOG_DIR)
    elif args.stage == "all":
        latest_path = run_stage_heuristics(args)
        if args.use_classifier:
            latest_path = run_stage_classifier(args, input_path=latest_path)
        if args.use_llm:
            latest_path = run_stage_llm(args, input_path=latest_path)
        run_stage_finalize(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

if __name__ == "__main__":
    main()
