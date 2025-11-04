# Clean Simple Wikipedia for Semantic Parsing

Extracts and cleans Simple Wikipedia sentences, outputting **one clean sentence per record** ready for semantic parsing.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test on sample (~100 documents)
python clean_simple_wiki.py --stage heuristics --test-limit 100
python generate_report.py heuristics  # Review in browser
python view_rejections.py              # Quick CLI review

# Finalize if satisfied
python clean_simple_wiki.py --stage finalize --test-limit 100
python generate_report.py final

# Run on full dataset (~770k articles → ~2-5M sentences)
python clean_simple_wiki.py --stage heuristics
python clean_simple_wiki.py --stage finalize
```

## 📋 Step-by-Step Processing Guide

### Step 1: Run Heuristics Stage

Extract and filter sentences using rule-based heuristics:

```bash
# Test mode (100 documents)
python clean_simple_wiki.py --stage heuristics --test-limit 100
```

**What it does:**
- Extracts sentences from Wikipedia text
- Removes wiki markup (links, citations, formatting)
- Filters out non-sentences (headings, lists, tables, fragments)
- Logs every accept/reject decision

**Output:**
- `simple_wiki_clean_heuristics.parquet` - accepted sentences
- `logs/01_heuristics.jsonl` - detailed decisions

---

### Step 2: Review Heuristics Results

**Option A: HTML Report (Visual, Comprehensive)**
```bash
python generate_report.py heuristics
# Opens: report_heuristics.html in browser
```

Shows:
- ✅ Acceptance rate (should be ~85-95%)
- ❌ Rejection breakdown by reason
- 📊 Visual charts
- 📝 Example sentences (accepted & rejected)

**Option B: Command Line (Quick)**
```bash
# View all rejections
python view_rejections.py

# Filter by specific reason
python view_rejections.py heuristics --reason length
python view_rejections.py heuristics --reason not_sentence_like
```

**Option C: Spot Check (Samples Only)**
```bash
python spot_check.py heuristics
```

**What to look for:**
- Are rejection reasons correct? (too short, not sentence-like, etc.)
- Are valid sentences being rejected? → Lower thresholds
- Are invalid sentences passing? → Raise thresholds
- Target rejection rate: 5-15%

---

### Step 3: Adjust Thresholds (If Needed)

If filtering is too aggressive or lenient, edit `clean_simple_wiki.py`:

```python
# Line 100: Minimum characters (default: 15)
if len(s) < 15:  # Lower to 10 for more recall

# Line 112: Minimum words (default: 3)  
if len(s.split()) < 3:  # Lower to 2 for more recall

# Line 115: Punctuation requirement (default: 8 words)
if len(s.split()) < 8 and not s.rstrip().endswith(('.', '!', '?')):
    # Raise 8 → 10 to be stricter
```

Then rerun:
```bash
rm simple_wiki_clean_heuristics.parquet logs/*
python clean_simple_wiki.py --stage heuristics --test-limit 100
python generate_report.py heuristics  # Review again
```

---

### Step 4: Optional Quality Stages

**Add Classifier (Optional)**
```bash
python clean_simple_wiki.py --stage classifier --use_classifier --test-limit 100
python generate_report.py classifier
python view_rejections.py classifier
```

Scores each sentence for quality (0-1). Useful if you want quantitative filtering.

**Add LLM Review (Optional)**
```bash
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage llm --use_llm --test-limit 100
python generate_report.py llm
python view_rejections.py llm
```

Uses GPT-4o-mini to judge borderline sentences. Shows what was kept/dropped and why.

**If LLM processing fails:** The script now preserves failed batches in checkpoints and creates detailed error logs. Use:
```bash
python clean_simple_wiki.py --stage analyze-failures  # Analyze what failed
```

**Skip these if:** Heuristics alone are sufficient (recommended for semantic parsing).

---

### Step 5: Finalize Output

Create final clean dataset:

```bash
python clean_simple_wiki.py --stage finalize --test-limit 100
```

**Output:**
- `simple_wiki_clean.jsonl` - one sentence per line (newline-delimited JSON)
- `simple_wiki_clean.parquet` - same data, columnar format

---

### Step 6: Review Final Output

```bash
# HTML report
python generate_report.py final

# Quick samples
head -20 simple_wiki_clean.jsonl

# Count sentences
wc -l simple_wiki_clean.jsonl
```

**Verify:**
- Sentences are clean (no wiki markup)
- Each record is a complete sentence
- Metadata is present (title, source_idx, etc.)

---

### Step 7: Run Full Dataset

Once satisfied with test results:

```bash
# Remove --test-limit flag to process all 770k articles
python clean_simple_wiki.py --stage heuristics
python clean_simple_wiki.py --stage finalize

# Expected output: 2-5 million sentences
# Time: 30-60 minutes
# Size: ~500MB-1GB
```

---

## 🔍 Review Tools Summary

| Tool | Purpose | Output |
|------|---------|--------|
| `generate_report.py heuristics` | Visual report with charts | HTML file in browser |
| `view_rejections.py` | Quick CLI view of rejections | Terminal output |
| `view_rejections.py all` | All pipeline stages at once | Terminal output |
| `spot_check.py` | Random samples | Terminal output |

**Examples:**
```bash
# See what was filtered and why
python view_rejections.py heuristics

# See only length rejections
python view_rejections.py heuristics --reason length

# Review classifier drops
python view_rejections.py classifier

# Review LLM decisions
python view_rejections.py llm

# See everything
python view_rejections.py all
```

## What It Does

1. **Extracts sentences** from Simple Wikipedia articles
2. **Removes wiki markup** (links, citations, formatting)
3. **Filters out** non-sentences (headings, lists, tables, fragments)
4. **Outputs** one clean sentence per record in JSONL/Parquet format

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

Perfect for feeding directly to semantic parsers!

## Pipeline Stages

### Stage 1: Heuristics (Required)
Extracts and filters sentences using rule-based heuristics.

```bash
python clean_simple_wiki.py --stage heuristics --test-limit 100
```

**Filtering rules:**
- Minimum 3 words, 15 characters
- Maximum 1000 characters
- Must contain letters
- No wiki markup (headings, lists, tables)
- Short sentences (<8 words) must end with `.!?`
- Uses NLTK for proper sentence splitting (handles abbreviations)

**Output**: `simple_wiki_clean_heuristics.parquet`

### Stage 2: Classifier (Optional)
Scores sentence quality using zero-shot classification.

```bash
python clean_simple_wiki.py --stage classifier --use_classifier --test-limit 100
```

**When to use:**
- You want quantitative quality scores
- You want to filter borderline cases
- You plan to use LLM for gray-zone sentences

**Skip if:** Heuristics are sufficient (recommended for semantic parsing)

### Stage 3: LLM (Optional)
Uses OpenAI to judge gray-zone sentences flagged by the classifier.

**Online mode (default, good for spot checks / <10k sentences):**

```bash
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage llm --use_llm --llm_mode online --test-limit 100
```

**Batch mode (recommended for ~770k sentences / production runs):**

1. **Prepare batch payloads** (writes JSONL payloads + manifest under `logs/batches/<run_name>`)
   ```bash
   python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_prepare --batch_run_name simplewiki_v1
   ```
   - Automatically shards gray-zone sentences into request files
   - Saves `pending_sentences.parquet` for deterministic replay
   - Omit `--batch_run_name` to auto-generate a timestamped directory; submit/collect default to the most recent run

2. **Submit jobs to OpenAI Batch API** (one batch per request file)
   ```bash
   python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_submit --batch_run_name simplewiki_v1
   ```
   - Uploads each JSONL file (`purpose="batch"`) and creates a `/v1/chat/completions` batch
   - Records `batch_id`, `input_file_id`, `status`, etc. in `manifest.json`

3. **Collect + merge results** (idempotent—run until all batches are `completed`)
   ```bash
   python clean_simple_wiki.py --stage llm --use_llm --llm_mode batch_collect --batch_run_name simplewiki_v1
   ```
   - Downloads output JSONL from OpenAI file storage
   - Parses `choices[0].message.content` via `parse_llm_response_text`
   - Applies results to the parquet dataset, updates checkpoints/logs, marks `llm_pending=False`

**Batch directory layout (`logs/batches/<run_name>`):**

```
meta/manifest.json           # per-request metadata + batch ids
meta/pending_sentences.parquet
requests/requests_00001.jsonl  # JSONL payloads for OpenAI Batch API
responses/<batch_id>_output.jsonl
```

**When to use:**
- You have OpenAI API access and want maximum quality (precision over recall)
- You need to process hundreds of thousands of sentences without hitting RPM/TPM limits

**Skip if:** No API key or budget concerns

### Stage 4: Finalize (Required)
Creates final clean output.

```bash
python clean_simple_wiki.py --stage finalize --test-limit 100
```

**Output**: 
- `simple_wiki_clean.jsonl` (newline-delimited JSON)
- `simple_wiki_clean.parquet` (columnar format)

## 📊 Understanding the Output

### Test Mode Results (100 documents)

**Typical numbers:**
- Input documents: 100
- Documents with accepted sentences: 85
- Documents fully filtered: 15 (no valid sentences)
- Total sentence candidates: ~250-300
- Accepted sentences: ~240 (93-95%)
- Rejected sentences: ~15 (5-7%)

**Why 85 documents vs 100?**

Some documents are completely filtered out because they contain:
- Only the title (e.g., "April", "Art")
- Only lists or formatting (e.g., numbered items)
- No valid sentences after wiki markup removal

This is **correct behavior** - these documents don't contribute sentences for semantic parsing.

---

## 📈 Common Rejection Reasons

| Reason | Description | Example |
|--------|-------------|---------|
| `length` | Too short (< 15 chars) or too long (> 1000 chars) | "April" |
| `not_sentence_like` | Missing punctuation, incomplete | "The functions include:" |
| `too_few_words` | Less than 3 words | "Alanis Morissette" |
| `heading` | Wiki heading format | "== Introduction ==" |
| `list` | List item marker | "* First item" |
| `table` | Wiki table syntax | "\| cell \|" |

View examples of each:
```bash
python view_rejections.py heuristics --reason length
python view_rejections.py heuristics --reason not_sentence_like
```

## All-in-One Command

Run everything at once:

```bash
# Minimal (heuristics + finalize)
python clean_simple_wiki.py --stage all --test-limit 100

# With classifier
python clean_simple_wiki.py --stage all --use_classifier --test-limit 100

# With everything (requires OPENAI_API_KEY)
python clean_simple_wiki.py --stage all --use_classifier --use_llm --test-limit 100
```

## Output Files

**Test mode (~100 docs)**:
- `simple_wiki_clean_heuristics.parquet`: ~60KB, ~200-300 sentences
- `logs/01_heuristics.jsonl`: ~30KB, detailed decisions
- `simple_wiki_clean.jsonl`: Final output

**Full dataset (~770k docs)**:
- Expected: 2-5 million clean sentences
- Size: ~500MB-1GB
- Time: 30-60 minutes on modern hardware

## ⚙️ Advanced Options

```bash
# Custom test limit (default: no limit for full processing)
python clean_simple_wiki.py --stage heuristics --test-limit 50   # Process 50 records
python clean_simple_wiki.py --stage heuristics --test-limit 500  # Process 500 records

# Custom output directory
python clean_simple_wiki.py --stage heuristics \
    --output_dir ./data \
    --out_prefix wiki_clean

# Custom log directory
python clean_simple_wiki.py --stage heuristics \
    --log_dir ./my_logs

# Adjust processing speed (only if logs not needed)
python clean_simple_wiki.py --stage heuristics \
    --num_proc 4  # Default: 1 (ensures logs work correctly)
    
# Note: num_proc > 1 may cause incomplete logs due to race conditions
```

## Dataset Info

- **Source**: [rahular/simple-wikipedia](https://huggingface.co/datasets/rahular/simple-wikipedia)
- **Articles**: 769,764
- **Words**: 23,886,673
- **License**: CC BY-SA 3.0

## Files Structure

```
clean_simple_wiki/
├── clean_simple_wiki.py          # Main pipeline
├── generate_report.py            # HTML report generator
├── view_rejections.py            # CLI rejection viewer (NEW)
├── spot_check.py                 # Terminal review tool
├── requirements.txt              # Dependencies
├── README.md                     # This file (quick reference)
├── TESTING_GUIDE.md             # Detailed testing instructions
│
├── simple_wiki_clean_heuristics.parquet  # Stage outputs
├── simple_wiki_clean.jsonl               # Final output
├── simple_wiki_clean.parquet             # Final output
│
└── logs/                         # Detailed decision logs
    ├── 01_heuristics.jsonl      # Every accept/reject with reason
    ├── 02_classifier.jsonl      # Sample quality scores
    ├── 03_llm.jsonl             # LLM decisions (if used)
    └── 04_final_decisions.jsonl # Final kept sentences
```

## Requirements

- Python 3.8+
- ~2GB RAM for test mode
- ~8GB RAM for full dataset
- Optional: OpenAI API key (for LLM stage)

## 🔧 Troubleshooting

**No logs generated?**
```bash
# Logs are created by default now (num_proc=1)
# If still missing:
python clean_simple_wiki.py --stage heuristics --test-limit 100 --num_proc 1
```

**Want to see what was rejected?**
```bash
# Quick view
python view_rejections.py

# Detailed HTML report
python generate_report.py heuristics
```

**Out of memory?**
```bash
# Already using minimal memory (num_proc=1 by default)
# If still issues, process fewer documents or increase RAM
```

**Too aggressive filtering?**
```bash
# 1. Check what's being rejected
python view_rejections.py heuristics

# 2. View rejection breakdown
python generate_report.py heuristics  # See charts

# 3. Adjust thresholds in clean_simple_wiki.py (see step 3 above)

# 4. Rerun and review
python clean_simple_wiki.py --stage heuristics --test-limit 100
python view_rejections.py
```

**Too lenient filtering?**
```bash
# Check final output quality
python generate_report.py final

# Raise thresholds or add classifier/LLM stages
python clean_simple_wiki.py --stage classifier --use_classifier --test-limit 100
```

**Need more details?**
- See `TESTING_GUIDE.md` for comprehensive instructions
- Check logs in `logs/` directory with `view_rejections.py`
- Open HTML reports for visual analysis

## 🎯 Quick Reference

**Processing Commands:**
```bash
# Test mode (100 docs)
python clean_simple_wiki.py --stage heuristics --test-limit 100
python clean_simple_wiki.py --stage finalize --test-limit 100

# Full dataset (770k docs)
python clean_simple_wiki.py --stage heuristics
python clean_simple_wiki.py --stage finalize
```

**Review Commands:**
```bash
# Visual reports (opens in browser)
python generate_report.py heuristics
python generate_report.py final

# CLI quick view
python view_rejections.py
python view_rejections.py all
python view_rejections.py heuristics --reason length

# Random samples
python spot_check.py
```

**Common Workflows:**

*Minimal (just heuristics):*
```bash
python clean_simple_wiki.py --stage all --test-limit 100
```

*With quality filtering:*
```bash
python clean_simple_wiki.py --stage all --use_classifier --test-limit 100
```

*Maximum quality (needs OpenAI key):*
```bash
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage all --use_classifier --use_llm --test-limit 100
```

---

## 📚 Next Steps

1. **Test**: Run on sample data (100 docs)
2. **Review**: Check `report_heuristics.html` and use `view_rejections.py`
3. **Adjust**: Tune thresholds if needed (typically not necessary)
4. **Finalize**: Create clean output with `--stage finalize`
5. **Scale**: Run on full dataset (remove `--test-limit`)
6. **Use**: Feed `simple_wiki_clean.jsonl` to your semantic parser

For detailed testing instructions, see **`TESTING_GUIDE.md`**.

## Citation

If you use this pipeline, please cite the Simple Wikipedia dataset:

```bibtex
@misc{simple-wikipedia,
  title={Simple Wikipedia},
  author={Rahul Aralikatte},
  year={2021},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/rahular/simple-wikipedia}
}
```
