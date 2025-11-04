# Testing Guide for Simple Wiki Pipeline

## Overview

For semantic parsing, you primarily need **heuristics + finalize**. The classifier and LLM stages are optional quality filters.

## Quick Test (Recommended Start)

```bash
# 1. Test heuristics (extracts & filters sentences)
python clean_simple_wiki.py --stage heuristics --test

# 2. Review heuristics results
python generate_report.py heuristics
# Opens: report_heuristics.html

# 3. Finalize to get clean output
python clean_simple_wiki.py --stage finalize --test

# 4. Review final output
python generate_report.py final
# Opens: report_final.html
```

## Stage-by-Stage Testing

### Stage 1: Heuristics (REQUIRED)
**Purpose**: Extract sentences, remove wiki markup, filter out non-sentences

```bash
# Test mode (~100 source documents)
python clean_simple_wiki.py --stage heuristics --test

# Output files:
# - simple_wiki_clean_heuristics.parquet (accepted sentences, 1 per record)
# - logs/01_heuristics.jsonl (detailed accept/reject decisions)

# Review results:
python generate_report.py heuristics

# Check what got filtered:
# - Rejection rate: Should be 5-15%
# - Common rejections: not_sentence_like, length, too_few_words
# - If too aggressive: adjust thresholds in clean_simple_wiki.py lines 100-115
```

**What to look for:**
- ✅ Clean sentences without wiki markup
- ✅ No headings (= Introduction =)
- ✅ No list markers (* item, 1. item)
- ✅ No fragments ("The capital.")
- ❌ If valid sentences rejected: lower word count threshold (line 112)

---

### Stage 2: Classifier (OPTIONAL)
**Purpose**: Score sentence quality, identify borderline cases

```bash
# Only needed if you want additional quality filtering
python clean_simple_wiki.py --stage classifier --use_classifier --test

# Output: simple_wiki_clean_classifier.parquet
# Columns: sentence + score_classifier + decision_source

# Review:
python spot_check.py classifier

# Thresholds (in clean_simple_wiki.py):
# - KEEP_THRESHOLD = 0.75  (high quality)
# - DROP_THRESHOLD = 0.35  (low quality)
# - Between: "gray" → sent to LLM (if enabled)
```

**When to use:**
- You want quantitative quality scores
- You want to filter out borderline cases
- You plan to use LLM for gray-zone sentences

**Skip if:**
- Heuristics are good enough
- You'll filter with semantic parser anyway
- You want maximum recall

---

### Stage 3: LLM (OPTIONAL)
**Purpose**: Human-like judgment on gray-zone sentences

```bash
# Only works with classifier enabled
export OPENAI_API_KEY=your_key_here
python clean_simple_wiki.py --stage llm --use_llm --test

# Output: simple_wiki_clean_llm.parquet
# Logs: logs/03_llm.jsonl (decisions + confidence)

# Review:
python spot_check.py llm
```

**When to use:**
- You have OpenAI API access
- You want highest quality (precision over recall)
- Classifier gray zone is large

**Skip if:**
- No API key / budget concerns
- Heuristics + classifier are sufficient
- You prefer deterministic filtering

---

### Stage 4: Finalize (REQUIRED)
**Purpose**: Create final clean output ready for semantic parsing

```bash
# Finalize whatever stages you ran
python clean_simple_wiki.py --stage finalize --test

# Output files:
# - simple_wiki_clean.jsonl (one sentence per line)
# - simple_wiki_clean.parquet (same data, columnar format)

# Review:
python generate_report.py final
# or
python spot_check.py final

# Check output format:
head -3 simple_wiki_clean.jsonl
```

**Output format:**
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

---

## Testing Workflows

### Minimal (Recommended for Semantic Parsing)
```bash
# Just heuristics + finalize
python clean_simple_wiki.py --stage heuristics --test
python generate_report.py heuristics
# Review, adjust thresholds if needed, then:
python clean_simple_wiki.py --stage finalize --test
python generate_report.py final
```

### With Quality Filtering
```bash
# Heuristics + classifier + finalize
python clean_simple_wiki.py --stage heuristics --test
python clean_simple_wiki.py --stage classifier --use_classifier --test
python clean_simple_wiki.py --stage finalize --test
python generate_report.py final
```

### Maximum Quality (with LLM)
```bash
# All stages
export OPENAI_API_KEY=...
python clean_simple_wiki.py --stage all --use_classifier --use_llm --test
python generate_report.py final
```

---

## Full Dataset Run

After testing, run on full dataset:

```bash
# Remove --test flag
python clean_simple_wiki.py --stage heuristics
python clean_simple_wiki.py --stage finalize

# Or all at once:
python clean_simple_wiki.py --stage all

# Expected:
# - Input: 769,764 Wikipedia articles
# - Output: ~2-5 million clean sentences
# - Time: 30-60 minutes
# - Size: ~500MB-1GB
```

---

## Adjusting Thresholds

Edit `clean_simple_wiki.py` if filtering is too aggressive/lenient:

```python
# Line 100: Minimum character length
if len(s) < 15:  # Try 10 for more recall

# Line 112: Minimum words
if len(s.split()) < 3:  # Try 2 for more recall

# Line 102: Maximum character length
if len(s) > MAX_SENT_CHARS:  # MAX_SENT_CHARS = 1000

# Line 115: Punctuation requirement
if len(s.split()) < 8 and not s.rstrip().endswith(('.', '!', '?')):
    # Adjust threshold 8 → 10 to be stricter
```

After changes, rerun:
```bash
rm simple_wiki_clean_heuristics.parquet logs/*
python clean_simple_wiki.py --stage heuristics --test
python generate_report.py heuristics
```

---

## Interpreting Reports

### `report_heuristics.html`
- **Acceptance rate**: Should be 85-95%
- **Rejection examples**: Verify they're actually bad
- **Sample sentences**: Should look clean and complete

### `report_final.html`
- **Total sentences**: Final count
- **Sample sentences**: Ready for semantic parsing?
- **Decision source**: Which stage made the decision

---

## Troubleshooting

**Too many sentences rejected?**
- Lower word count threshold (line 112)
- Lower character minimum (line 100)
- Check rejection examples in report

**Low quality sentences passing?**
- Raise word count threshold
- Add classifier stage
- Check if semantic parser can handle it

**No logs generated?**
- `logs/` directory should exist (auto-created)
- Check file permissions
- Verify the pipeline completed successfully

**OOM on full dataset?**
- Default is `--num_proc 1` (can increase if you have RAM and don't need logs)
- Process in batches manually
- Increase system RAM

---

## Next Steps

1. **Test on sample** → Review → Adjust
2. **Run full dataset** → Takes 30-60 min
3. **Feed to semantic parser** → Use `simple_wiki_clean.jsonl`
4. **Iterate** based on parsing results

