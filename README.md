# Clean Simple Wikipedia

A modular pipeline that extracts and cleans Simple Wikipedia sentences for semantic parsing. Transforms raw Wikipedia articles into clean, standalone sentences ready for downstream NLP tasks.

## What It Does

- **Extracts** sentences from 770k Simple Wikipedia articles
- **Removes** wiki markup (links, citations, formatting)
- **Filters** out non-sentences (headings, lists, tables, fragments)
- **Outputs** clean sentences in JSONL/Parquet format (one sentence per record)

## Architecture

```
src/
├── config.py                      # Configuration constants
├── cli.py                         # Command-line interface
├── text_processing/               # Wiki markup cleaning & sentence extraction
├── models/                        # ML classifier & LLM processor
├── batch_processing/              # OpenAI batch API management
├── stages/                        # Processing pipeline stages
└── utils/                         # Logging, I/O, data utilities
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test on 100 documents
python clean_simple_wiki.py --stage all --test-limit 100

# Review results
python generate_report.py final

# Run on full dataset (770k articles → 2-5M sentences)
python clean_simple_wiki.py --stage all
```

## Output Format

**File**: `simple_wiki_clean.jsonl` (one sentence per line)

```json
{
  "row_id": 0,
  "title": "April",
  "source_idx": 1,
  "sentence_idx": 0,
  "split": "train",
  "sentence": "April is the fourth month of the year.",
  "decision_source": "heuristics"
}
```

## Processing Stages

The pipeline has 4 stages that can be run individually or together:

1. **Heuristics** (required) - Rule-based filtering
2. **Classifier** (optional) - ML quality scoring  
3. **LLM** (optional) - AI review of borderline cases
4. **Finalize** (required) - Generate clean output

### Stage Options

```bash
# Minimal (heuristics only)
python clean_simple_wiki.py --stage all

# With ML classifier
python clean_simple_wiki.py --stage all --use_classifier

# With AI review (requires OpenAI API key)
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage all --use_classifier --use_llm
```

## Review Tools

```bash
# Visual HTML reports
python generate_report.py heuristics
python generate_report.py final

# Command-line rejection viewer
python view_rejections.py
python view_rejections.py heuristics --reason length

# Random samples
python spot_check.py
```

## Configuration

Key settings in `src/config.py`:

```python
# Sentence validation thresholds
MAX_SENT_CHARS = 1000          # Maximum sentence length
KEEP_THRESHOLD = 0.75          # Classifier auto-keep threshold
DROP_THRESHOLD = 0.35          # Classifier auto-drop threshold
```

Validation rules in `src/text_processing/validation.py`:

```python
# Minimum requirements
if len(s) < 15: return False, "length"           # Min 15 characters
if len(s.split()) < 3: return False, "too_few_words"  # Min 3 words
```

## Requirements

- Python 3.8+
- ~2GB RAM (test mode) / ~8GB RAM (full dataset)
- Optional: OpenAI API key for LLM stage

## Dataset

- **Source**: [rahular/simple-wikipedia](https://huggingface.co/datasets/rahular/simple-wikipedia)
- **Articles**: 769,764
- **Expected Output**: 2-5 million clean sentences
- **Processing Time**: 30-60 minutes (full dataset)

---

# Usage Guide

## Testing Workflow

### 1. Run Test Processing

```bash
# Process 100 documents for testing
python clean_simple_wiki.py --stage all --test-limit 100
```

This runs all stages and produces:
- `simple_wiki_clean.jsonl` - Final clean sentences
- `logs/` - Detailed processing logs

### 2. Review Results

**Visual Report (Recommended)**
```bash
python generate_report.py final
# Opens report_final.html in browser
```

**Command Line Review**
```bash
# See what was filtered out and why
python view_rejections.py

# Check specific rejection reasons
python view_rejections.py heuristics --reason length
python view_rejections.py heuristics --reason not_sentence_like
```

**Quick Sample Check**
```bash
# View random samples
python spot_check.py

# Check final output format
head -10 simple_wiki_clean.jsonl
```

### 3. Adjust If Needed

**Too many sentences rejected?**
Edit `src/text_processing/validation.py`:
```python
# Lower thresholds for more recall
if len(s) < 10:  # was 15
if len(s.split()) < 2:  # was 3
```

**Poor quality sentences passing?**
Edit `src/config.py`:
```python
# Raise thresholds for higher precision
KEEP_THRESHOLD = 0.80  # was 0.75
DROP_THRESHOLD = 0.40  # was 0.35
```

Then rerun:
```bash
rm simple_wiki_clean*.parquet logs/*
python clean_simple_wiki.py --stage all --test-limit 100
```

### 4. Run Full Dataset

Once satisfied with test results:

```bash
# Remove --test-limit to process all 770k articles
python clean_simple_wiki.py --stage all

# Expected: 2-5 million sentences, 30-60 minutes
```

## Advanced Usage

### Individual Stages

```bash
# Run stages separately
python clean_simple_wiki.py --stage heuristics --test-limit 100
python clean_simple_wiki.py --stage classifier --use_classifier --test-limit 100
python clean_simple_wiki.py --stage llm --use_llm --test-limit 100
python clean_simple_wiki.py --stage finalize --test-limit 100
```

### LLM Batch Processing

For large datasets, use batch mode to avoid rate limits:

```bash
# 1. Prepare batch requests
python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_prepare

# 2. Submit to OpenAI
python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_submit

# 3. Collect results (run periodically until complete)
python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_collect
```

### Custom Configuration

```bash
# Custom output directory
python clean_simple_wiki.py --stage all --output_dir ./data --out_prefix wiki_clean

# Custom log directory  
python clean_simple_wiki.py --stage all --log_dir ./my_logs

# Process specific number of documents
python clean_simple_wiki.py --stage all --test-limit 500
```

## Understanding the Output

### Typical Results (100 test documents)

- **Input documents**: 100
- **Documents with sentences**: ~85 (some have no valid sentences)
- **Total sentence candidates**: ~250-300
- **Final clean sentences**: ~240 (90-95% acceptance rate)

### Common Rejection Reasons

| Reason | Description | Example |
|--------|-------------|---------|
| `length` | Too short/long | "April" |
| `not_sentence_like` | Missing punctuation | "The functions include:" |
| `too_few_words` | Less than 3 words | "Alanis Morissette" |
| `heading` | Wiki heading | "== Introduction ==" |
| `list` | List item | "* First item" |
| `table` | Table markup | "\| cell \|" |

### Quality Stages Explained

**Heuristics** (required): Fast rule-based filtering
- Removes obvious non-sentences (headings, lists, fragments)
- ~90-95% of sentences pass

**Classifier** (optional): ML quality scoring
- Scores sentences 0-1 for quality
- Auto-keeps high scores (>0.75), auto-drops low scores (<0.35)
- Sends borderline scores (0.35-0.75) to LLM if enabled

**LLM** (optional): AI review of borderline cases
- Only processes ~10-30% of sentences (the "gray zone")
- Provides human-like judgment on difficult cases
- Costs ~$100-300 for full dataset

## Troubleshooting

**No output files?**
```bash
# Check if processing completed successfully
ls -la simple_wiki_clean*
```

**Want to see what was filtered?**
```bash
python view_rejections.py
python generate_report.py heuristics
```

**Out of memory?**
```bash
# Already optimized for memory (num_proc=1)
# Try processing fewer documents or add more RAM
python clean_simple_wiki.py --stage all --test-limit 50
```

**API errors with LLM stage?**
```bash
# Check failed batches
python clean_simple_wiki.py --stage analyze-failures

# Resume from checkpoint
python clean_simple_wiki.py --stage llm --use_llm
```

## Files Created

```
simple_wiki_clean.jsonl              # Final output (JSONL format)
simple_wiki_clean.parquet            # Final output (Parquet format)
simple_wiki_clean_heuristics.parquet # Intermediate stage outputs
simple_wiki_clean_classifier.parquet
simple_wiki_clean_llm.parquet
logs/
├── 01_heuristics.jsonl              # Detailed processing logs
├── 02_classifier.jsonl
├── 03_llm.jsonl
└── 04_final_decisions.jsonl
```

## Citation

```bibtex
@misc{simple-wikipedia,
  title={Simple Wikipedia},
  author={Rahul Aralikatte},
  year={2021},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/rahular/simple-wikipedia}
}
```