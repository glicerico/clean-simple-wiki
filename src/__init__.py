"""
Clean Simple Wikipedia - Modular text processing pipeline.

This package provides a staged processing pipeline for cleaning Simple Wikipedia data:
  heuristics → classifier → llm → finalize

Each stage is re-runnable and produces auditable logs.
"""

__version__ = "1.0.0"
