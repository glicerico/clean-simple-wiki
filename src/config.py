"""Configuration constants and settings for the clean-simple-wiki pipeline."""

# Dataset and output configuration
DEFAULT_DATASET = "rahular/simple-wikipedia"
OUT_PREFIX = "simple_wiki_clean"
DEFAULT_JSONL = f"{OUT_PREFIX}.jsonl"
DEFAULT_PARQUET = f"{OUT_PREFIX}.parquet"
LOG_DIR = "logs"

# Processing configuration
DEFAULT_NUM_PROC = 1  # Keep at 1 to ensure logs are written correctly
DEFAULT_BATCH_SIZE_CLASSIFIER = 64
MAX_SENT_CHARS = 1000

# Classifier thresholds
KEEP_THRESHOLD = 0.75
DROP_THRESHOLD = 0.35

# LLM configuration
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LLM_BATCH_SIZE = 16  # Conservative batch size for structured JSON output with coreference resolution
LLM_MAX_RETRIES = 1
LLM_TIMEOUT = 30  # Avoid hanging when no response from LLM

# Batch processing configuration
DEFAULT_LLM_MODE = "online"
DEFAULT_BATCH_DIR_NAME = "batches"
DEFAULT_BATCH_REQUESTS_PER_FILE = 500
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"

# Testing configuration
DEFAULT_TEST_LIMIT = 100

# Column name candidates for dataset parsing
TEXT_COL_CANDIDATES = ["text", "article", "content", "body"]
TITLE_COL_CANDIDATES = ["title", "heading"]

# Batch processing file names
BATCH_MANIFEST_FILENAME = "manifest.json"
BATCH_PENDING_PARQUET = "pending_sentences.parquet"
BATCH_RESPONSES_SUBDIR = "responses"
BATCH_REQUESTS_SUBDIR = "requests"
BATCH_META_SUBDIR = "meta"
