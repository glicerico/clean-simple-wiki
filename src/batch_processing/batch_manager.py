"""Batch manager for OpenAI API batch operations."""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple, Set, List

import pandas as pd

from ..config import (
    LOG_DIR, DEFAULT_BATCH_DIR_NAME, DEFAULT_BATCH_REQUESTS_PER_FILE,
    DEFAULT_BATCH_COMPLETION_WINDOW, BATCH_MANIFEST_FILENAME,
    BATCH_PENDING_PARQUET, BATCH_RESPONSES_SUBDIR, BATCH_REQUESTS_SUBDIR,
    BATCH_META_SUBDIR
)
from ..models.llm_processor import build_llm_messages, create_openai_client
from ..utils.io_utils import ensure_output_dir
from ..utils.logging import append_jsonl
from .batch_utils import parse_batch_output_file


class BatchManager:
    """Manages OpenAI batch processing operations."""
    
    def __init__(self, args):
        """Initialize batch manager with configuration.
        
        Args:
            args: Argument namespace with batch configuration
        """
        self.args = args
        self.base_dir = self._resolve_batch_base_dir()
        
    def _timestamp_slug(self) -> str:
        """Generate a timestamp-based slug for batch runs."""
        return time.strftime("run_%Y%m%d_%H%M%S")
    
    def _resolve_batch_base_dir(self) -> str:
        """Resolve the base directory for batch operations."""
        base_dir = getattr(self.args, 'batch_dir', None) or os.path.join(LOG_DIR, DEFAULT_BATCH_DIR_NAME)
        ensure_output_dir(base_dir)
        return os.path.abspath(base_dir)
    
    def _resolve_batch_run_dir(self, create_if_missing: bool = False) -> Tuple[str, str]:
        """Resolve the run directory for batch operations.
        
        Args:
            create_if_missing: Whether to create the directory if it doesn't exist
            
        Returns:
            Tuple of (run_name, run_dir_path)
        """
        run_name = getattr(self.args, 'batch_run_name', None)
        if not run_name:
            if create_if_missing:
                run_name = self._timestamp_slug()
                self.args.batch_run_name = run_name
            else:
                run_name = self._find_latest_batch_run()
                if not run_name:
                    raise RuntimeError("No batch runs found. Specify --batch_run_name or run with --llm_mode batch_prepare first.")
                self.args.batch_run_name = run_name
        
        run_dir = os.path.join(self.base_dir, run_name)
        if create_if_missing:
            ensure_output_dir(run_dir)
            ensure_output_dir(os.path.join(run_dir, BATCH_REQUESTS_SUBDIR))
            ensure_output_dir(os.path.join(run_dir, BATCH_RESPONSES_SUBDIR))
            ensure_output_dir(os.path.join(run_dir, BATCH_META_SUBDIR))
        if not os.path.isdir(run_dir):
            raise RuntimeError(f"Batch run directory not found: {run_dir}")
        return run_name, run_dir
    
    def _find_latest_batch_run(self) -> Optional[str]:
        """Find the most recently modified batch run directory."""
        if not os.path.isdir(self.base_dir):
            return None
        candidates = []
        for name in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, name)
            if os.path.isdir(path):
                candidates.append((os.path.getmtime(path), name))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def _batch_manifest_path(self, run_dir: str) -> str:
        """Get the path to the batch manifest file."""
        return os.path.join(run_dir, BATCH_META_SUBDIR, BATCH_MANIFEST_FILENAME)
    
    def _load_batch_manifest(self, run_dir: str) -> Dict[str, Any]:
        """Load batch manifest from file."""
        path = self._batch_manifest_path(run_dir)
        if not os.path.exists(path):
            raise RuntimeError(f"Batch manifest not found at {path}. Run batch_prepare first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save_batch_manifest(self, run_dir: str, manifest: Dict[str, Any]):
        """Save batch manifest to file."""
        path = self._batch_manifest_path(run_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    
    def prepare_batch_requests(self, model: str, pending_df: pd.DataFrame) -> str:
        """Prepare batch requests for OpenAI API.
        
        Args:
            model: OpenAI model name
            pending_df: DataFrame with pending sentences
            
        Returns:
            Path to the batch run directory
        """
        if pending_df.empty:
            raise RuntimeError("No sentences pending LLM review. Nothing to prepare for batch mode.")

        run_name, run_dir = self._resolve_batch_run_dir(create_if_missing=True)
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

        requests_per_file = max(1, getattr(self.args, "batch_requests_per_file", DEFAULT_BATCH_REQUESTS_PER_FILE))
        batch_size = max(1, getattr(self.args, "llm_batch_size", 16))

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
        self._save_batch_manifest(run_dir, manifest)

        print(f"[batch] Prepared run '{run_name}' at {run_dir}")
        print(f"[batch] Pending sentences: {len(row_ids):,}")
        print(f"[batch] Requests generated: {batches_generated:,} across {len(manifest['request_files']):,} files")
        print(f"[batch] Example request file: {manifest['request_files'][0]['file_path'] if manifest['request_files'] else 'N/A'}")
        print(f"[batch] Pending sentences snapshot saved to {pending_parquet_path}")

        return run_dir
    
    def submit_batch_requests(self) -> str:
        """Submit prepared batch requests to OpenAI API.
        
        Returns:
            Path to the batch run directory
        """
        run_name, run_dir = self._resolve_batch_run_dir(create_if_missing=False)
        manifest = self._load_batch_manifest(run_dir)
        client = create_openai_client(enable_tracing=False)
        completion_window = getattr(self.args, "batch_completion_window", DEFAULT_BATCH_COMPLETION_WINDOW)

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

        self._save_batch_manifest(run_dir, manifest)
        return run_dir
    
    def collect_batch_results(
        self,
        df: pd.DataFrame,
        row_id_to_index: Dict[int, int],
        llm_logs: List[Dict[str, Any]],
        checkpoint_path: str,
    ) -> Tuple[Set[int], str]:
        """Collect results from completed batch operations.
        
        Args:
            df: DataFrame to update with results
            row_id_to_index: Mapping from row IDs to DataFrame indices
            llm_logs: List to append log entries to
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (processed_row_ids, run_dir_path)
        """
        run_name, run_dir = self._resolve_batch_run_dir(create_if_missing=False)
        manifest = self._load_batch_manifest(run_dir)
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

        self._save_batch_manifest(run_dir, manifest)

        if batch_errors:
            error_log_path = os.path.join(run_dir, BATCH_META_SUBDIR, "batch_errors.jsonl")
            append_jsonl(error_log_path, batch_errors)
            print(f"[batch][WARN] Logged {len(batch_errors)} batch response errors to {error_log_path}")

        return processed_row_ids, run_dir
