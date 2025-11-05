#!/usr/bin/env python3
"""
Clean Simple Wikipedia with auditable, staged processing.

This script now uses a modular architecture. The main functionality has been
refactored into the src/ package for better organization and maintainability.

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

from src.cli import main


if __name__ == "__main__":
    main()