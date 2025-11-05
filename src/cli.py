"""Command-line interface for the clean-simple-wiki pipeline."""

import os
import argparse

from .config import (
    DEFAULT_DATASET, OUT_PREFIX, LOG_DIR, DEFAULT_NUM_PROC,
    DEFAULT_BATCH_SIZE_CLASSIFIER, DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_MODEL,
    LLM_BATCH_SIZE, DEFAULT_LLM_MODE, DEFAULT_BATCH_REQUESTS_PER_FILE,
    DEFAULT_BATCH_COMPLETION_WINDOW, DEFAULT_TEST_LIMIT
)
from .stages import (
    run_stage_heuristics, run_stage_classifier, run_stage_llm,
    run_stage_finalize, analyze_failed_batches
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    ap = argparse.ArgumentParser("Stageable Simple Wikipedia cleaner.")
    
    # Main execution options
    ap.add_argument("--stage", default="all", 
                    choices=["heuristics", "classifier", "llm", "finalize", "all", "analyze-failures"],
                    help="Processing stage to run")
    
    # Data and output configuration
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="HuggingFace dataset to process")
    ap.add_argument("--output_dir", default=".",
                    help="Output directory for processed files")
    ap.add_argument("--out_prefix", default=OUT_PREFIX,
                    help="Prefix for output files")
    ap.add_argument("--log_dir", default=LOG_DIR,
                    help="Directory for log files")
    ap.add_argument("--output_jsonl", default=None,
                    help="Custom path for final JSONL output")
    ap.add_argument("--output_parquet", default=None,
                    help="Custom path for final Parquet output")
    
    # Processing configuration
    ap.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC,
                    help="Number of processes for dataset mapping (default: 1 for reliable logging)")
    ap.add_argument("--test-limit", type=int, default=None,
                    help="Limit processing to N records for testing (applies to all stages)")
    
    # Classifier configuration
    ap.add_argument("--use_classifier", action="store_true",
                    help="When running --stage all, include classifier stage")
    ap.add_argument("--classifier_model", default="typeform/distilbert-base-uncased-mnli",
                    help="HuggingFace model for classification")
    ap.add_argument("--classifier_batch_size", type=int, default=DEFAULT_BATCH_SIZE_CLASSIFIER,
                    help="Batch size for classifier processing")
    
    # LLM configuration
    ap.add_argument("--use_llm", action="store_true",
                    help="When running --stage all, include LLM stage")
    ap.add_argument("--llm_provider", default=DEFAULT_LLM_PROVIDER, choices=["openai"],
                    help="LLM provider to use")
    ap.add_argument("--openai_model", default=DEFAULT_OPENAI_MODEL,
                    help="OpenAI model name")
    ap.add_argument("--llm_batch_size", type=int, default=LLM_BATCH_SIZE,
                    help="Batch size for LLM processing")
    ap.add_argument("--llm_mode", default=DEFAULT_LLM_MODE,
                    choices=["online", "batch_prepare", "batch_submit", "batch_collect"],
                    help="LLM execution mode")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore checkpoint and reprocess all LLM sentences")
    
    # Batch processing configuration
    ap.add_argument("--batch_dir", default=None,
                    help="Directory for OpenAI batch payloads and results (default: <log_dir>/batches)")
    ap.add_argument("--batch_run_name", default=None,
                    help="Identifier for OpenAI batch runs; default auto-generates a timestamp when preparing batches")
    ap.add_argument("--batch_requests_per_file", type=int, default=DEFAULT_BATCH_REQUESTS_PER_FILE,
                    help="Number of API requests to store per JSONL batch request file when preparing batches")
    ap.add_argument("--batch_completion_window", default=DEFAULT_BATCH_COMPLETION_WINDOW,
                    help="OpenAI batch completion window (e.g., '24h')")
    
    return ap


def process_arguments(args):
    """Process and validate command-line arguments.
    
    Args:
        args: Parsed argument namespace
    """
    # Convert paths to absolute
    args.output_dir = os.path.abspath(args.output_dir)
    if args.output_jsonl:
        args.output_jsonl = os.path.abspath(args.output_jsonl)
    if args.output_parquet:
        args.output_parquet = os.path.abspath(args.output_parquet)
    
    # Update global LOG_DIR
    global LOG_DIR
    from . import config
    config.LOG_DIR = os.path.abspath(args.log_dir)
    
    if args.batch_dir:
        args.batch_dir = os.path.abspath(args.batch_dir)


def run_pipeline(args):
    """Run the processing pipeline based on arguments.
    
    Args:
        args: Parsed and processed argument namespace
    """
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
        from . import config
        analyze_failed_batches(config.LOG_DIR)
    elif args.stage == "all":
        latest_path = run_stage_heuristics(args)
        if args.use_classifier:
            latest_path = run_stage_classifier(args, input_path=latest_path)
        if args.use_llm:
            latest_path = run_stage_llm(args, input_path=latest_path)
        run_stage_finalize(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    process_arguments(args)
    run_pipeline(args)
