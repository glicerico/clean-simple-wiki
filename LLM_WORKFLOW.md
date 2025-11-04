# LLM Workflow: Only Dubious Sentences

## How It Works

The pipeline is designed so that **only borderline/dubious sentences** reach the LLM, saving you time and API costs.

## Three-Tier Decision System

```
Heuristics → Classifier → LLM (only gray zone)
```

### Stage 1: Heuristics (Fast, Rule-Based)
Filters obvious bad content:
- ❌ Headings, lists, tables
- ❌ Too short (<3 words)
- ❌ Too long (>1000 chars)
- ❌ Missing punctuation
- ✅ Keeps: Valid-looking sentences

**Result**: ~85-95% of sentences pass (good recall)

### Stage 2: Classifier (Fast, ML-Based)
Scores sentence quality (0-1 scale):

```python
KEEP_THRESHOLD = 0.75   # High quality → auto-keep
DROP_THRESHOLD = 0.35   # Low quality → auto-drop
# Between 0.35-0.75 → Gray zone → needs LLM
```

**Example scores:**
- 0.92: "April is the fourth month of the year." → ✅ AUTO-KEEP
- 0.15: "Click here for more information" → ❌ AUTO-DROP
- 0.55: "Poets use April to mean end of winter" → ⚠️ DUBIOUS → send to LLM

### Stage 3: LLM (Slow, Expensive, Accurate)
**Only processes gray zone sentences** (score 0.35-0.75)

Reviews ~10-30% of sentences, not all of them.

## Example Run

```bash
# Test on 241 sentences:
python clean_simple_wiki.py --stage heuristics --test
# Result: 241 sentences pass heuristics

python clean_simple_wiki.py --stage classifier --use_classifier --test
# Result: 
#   - 34 high quality (>0.75) → KEEP
#   - 1 low quality (<0.35) → DROP
#   - 165 gray zone (0.35-0.75) → LLM_PENDING

# Only 165 sentences (68%) go to LLM, not all 241
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage llm --use_llm --test
# Result: LLM reviews only 165 dubious sentences
#   - ~100 kept
#   - ~65 dropped
```

## Cost Savings

**Without classifier** (sending all to LLM):
- 241 sentences × $0.0001 = $0.024

**With classifier** (sending only gray zone):
- 165 sentences × $0.0001 = $0.0165
- **Savings: 32%**

On full dataset (~2-5M sentences):
- Without: $200-500
- With: $130-340
- **Savings: ~$100-150**

## Adjusting the Gray Zone

Edit `clean_simple_wiki.py` lines 59-60:

```python
KEEP_THRESHOLD = 0.75  # Lower to 0.70 → more auto-keep, fewer LLM calls
DROP_THRESHOLD = 0.35  # Raise to 0.40 → more auto-drop, fewer LLM calls
```

**To minimize LLM calls** (higher precision, lower recall):
```python
KEEP_THRESHOLD = 0.70  # More lenient
DROP_THRESHOLD = 0.40  # More strict
# Result: Smaller gray zone, fewer LLM calls
```

**To maximize quality** (use LLM more):
```python
KEEP_THRESHOLD = 0.80  # More strict
DROP_THRESHOLD = 0.30  # More lenient
# Result: Larger gray zone, more LLM calls, higher quality
```

## Skipping LLM Entirely

If you don't want to use LLM at all:

```bash
# Run classifier only
python clean_simple_wiki.py --stage heuristics --test
python clean_simple_wiki.py --stage classifier --use_classifier --test
python clean_simple_wiki.py --stage finalize --test

# Gray zone sentences are dropped (not sent to LLM)
# Only high-confidence sentences (>0.75) are kept
```

This gives you high precision but lower recall.

## Full Workflow

```bash
# 1. Heuristics (required)
python clean_simple_wiki.py --stage heuristics

# 2. Classifier (marks dubious sentences)
python clean_simple_wiki.py --stage classifier --use_classifier

# 3. LLM (reviews only dubious sentences)
export OPENAI_API_KEY=your_key
python clean_simple_wiki.py --stage llm --use_llm

# 4. Finalize
python clean_simple_wiki.py --stage finalize

# Or all at once:
python clean_simple_wiki.py --stage all --use_classifier --use_llm
```

## Monitoring

Check how many sentences reach LLM:

```bash
python spot_check.py

# Output shows:
# [classifier] Gray zone (dubious, needs LLM review): X sentences
# [classifier] Score range for gray zone: 0.35 - 0.75
# [llm] Found X dubious sentences (Y% of total)
```

## Summary

✅ **Only borderline sentences go to LLM**
✅ **High-quality sentences auto-kept** (fast, free)
✅ **Low-quality sentences auto-dropped** (fast, free)
✅ **Gray zone reviewed by LLM** (slow, costs money, but accurate)
✅ **Saves 30-70% on API costs** vs sending everything to LLM

