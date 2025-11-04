#!/usr/bin/env python3
"""
Quick spot check script to review pipeline outputs.
Usage: python spot_check.py [stage_name]
  stage_name: heuristics, classifier, llm, or final (default: auto-detect latest)
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path

def check_heuristics():
    """Check heuristics stage output."""
    path = "simple_wiki_clean_heuristics.parquet"
    if not os.path.exists(path):
        print(f"❌ {path} not found. Run: python clean_simple_wiki.py --stage heuristics --test")
        return
    
    df = pd.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"HEURISTICS STAGE: {path}")
    print(f"{'='*60}")
    print(f"Total sentences: {len(df):,} (1 per record)")
    
    if "source_idx" in df.columns:
        num_sources = df["source_idx"].nunique()
        avg_per_source = len(df) / num_sources if num_sources > 0 else 0
        print(f"From {num_sources:,} source documents")
        print(f"Avg sentences per document: {avg_per_source:.1f}")
    
    print(f"\n{'='*60}")
    print("SAMPLE SENTENCES:")
    print(f"{'='*60}")
    samples = df.head(10)
    for idx, row in samples.iterrows():
        title_str = row['title'][:60] if row['title'] else "(no title)"
        sent_idx_str = f"[{row['sentence_idx']}]" if "sentence_idx" in row else ""
        print(f"\n{sent_idx_str} From: {title_str}")
        print(f"    {row['sentence'][:150]}...")
    
    # Check log file
    log_path = "logs/01_heuristics.jsonl"
    if os.path.exists(log_path):
        print(f"\n{'='*60}")
        print(f"FILTERING BREAKDOWN: {log_path}")
        print(f"{'='*60}")
        
        # Read all logs to get statistics
        all_logs = []
        with open(log_path, 'r') as f:
            for line in f:
                all_logs.append(json.loads(line))
        
        total_logs = len(all_logs)
        accepted = [l for l in all_logs if l['accepted']]
        rejected = [l for l in all_logs if not l['accepted']]
        
        print(f"\nTotal sentence candidates: {total_logs:,}")
        print(f"Accepted: {len(accepted):,} ({len(accepted)/total_logs*100:.1f}%)")
        print(f"Rejected: {len(rejected):,} ({len(rejected)/total_logs*100:.1f}%)")
        
        # Breakdown by rejection reason
        if rejected:
            from collections import Counter
            reasons = Counter(r['filter_reason'] for r in rejected)
            print(f"\nRejection reasons:")
            for reason, count in reasons.most_common():
                print(f"  {reason}: {count:,} ({count/len(rejected)*100:.1f}%)")
            
            print(f"\n{'='*60}")
            print("EXAMPLES OF FILTERED CONTENT:")
            print(f"{'='*60}")
            
            # Show examples for each major rejection reason
            shown_reasons = set()
            for r in rejected:
                reason = r['filter_reason']
                if reason not in shown_reasons and len(shown_reasons) < 5:
                    shown_reasons.add(reason)
                    print(f"\n❌ Reason: {reason}")
                    text = r['sentence_after'] if r['sentence_after'] else r['fragment_before']
                    print(f"   Text: {text[:120]}")
        
        if accepted:
            print(f"\n{'='*60}")
            print("EXAMPLES OF ACCEPTED CONTENT:")
            print(f"{'='*60}")
            for a in accepted[:3]:
                print(f"\n✅ Accepted")
                print(f"   Before: {a['fragment_before'][:100]}")
                print(f"   After:  {a['sentence_after'][:100]}")

def check_classifier():
    """Check classifier stage output."""
    path = "simple_wiki_clean_classifier.parquet"
    if not os.path.exists(path):
        print(f"❌ {path} not found. Run: python clean_simple_wiki.py --stage classifier --use_classifier --test")
        return
    
    df = pd.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"CLASSIFIER STAGE: {path}")
    print(f"{'='*60}")
    print(f"Total records: {len(df):,}")
    print(f"\nDecision breakdown:")
    print(df['decision_source'].value_counts().to_string())
    
    if 'score_classifier' in df.columns:
        scores = df[df['score_classifier'].notna()]['score_classifier']
        print(f"\nClassifier scores:")
        print(f"  Mean: {scores.mean():.3f}")
        print(f"  Median: {scores.median():.3f}")
        print(f"  Std: {scores.std():.3f}")
    
    print(f"\n{'='*60}")
    print("SAMPLE KEPT BY CLASSIFIER:")
    print(f"{'='*60}")
    kept = df[df['decision_source'] == 'classifier_keep'].head(2)
    for idx, row in kept.iterrows():
        print(f"\n✅ [{idx}] Score: {row.get('score_classifier', 'N/A'):.3f}")
        print(f"    Title: {row['title'][:60]}")
        print(f"    Content: {row['chunk_clean'][:150]}...")
    
    print(f"\n{'='*60}")
    print("SAMPLE DROPPED BY CLASSIFIER:")
    print(f"{'='*60}")
    dropped = df[df['decision_source'] == 'classifier_drop'].head(2)
    for idx, row in dropped.iterrows():
        print(f"\n❌ [{idx}] Score: {row.get('score_classifier', 'N/A'):.3f}")
        print(f"    Title: {row['title'][:60]}")
        print(f"    Content: {row['chunk_clean'][:150]}...")
    
    print(f"\n{'='*60}")
    print("SAMPLE BORDERLINE CASES (gray zone → needs LLM):")
    print(f"{'='*60}")
    gray = df[df['decision_source'] == 'classifier_gray'].head(2)
    for idx, row in gray.iterrows():
        print(f"\n⚠️  [{idx}] Score: {row.get('score_classifier', 'N/A'):.3f}")
        print(f"    Title: {row['title'][:60]}")
        print(f"    Content: {row['chunk_clean'][:150]}...")

def check_llm():
    """Check LLM stage output."""
    path = "simple_wiki_clean_llm.parquet"
    if not os.path.exists(path):
        print(f"❌ {path} not found. Run: python clean_simple_wiki.py --stage llm --use_llm --test")
        return
    
    df = pd.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"LLM STAGE: {path}")
    print(f"{'='*60}")
    print(f"Total records: {len(df):,}")
    print(f"\nDecision breakdown:")
    print(df['decision_source'].value_counts().to_string())
    
    llm_processed = df[df['decision_source'] == 'llm']
    if len(llm_processed) > 0:
        print(f"\nLLM processing:")
        print(f"  Kept: {llm_processed['keep'].sum():,}")
        print(f"  Dropped: {(~llm_processed['keep']).sum():,}")
        print(f"  Avg confidence: {llm_processed['confidence'].mean():.3f}")
    
    print(f"\n{'='*60}")
    print("SAMPLE LLM KEPT:")
    print(f"{'='*60}")
    kept_samples = llm_processed[llm_processed['keep']].head(2)
    for idx, row in kept_samples.iterrows():
        print(f"\n✅ [{idx}] Confidence: {row['confidence']:.2f}")
        print(f"    Before: {row['chunk_clean'][:100]}...")
        print(f"    After:  {row['cleaned'][:100]}...")
        changed = row['chunk_clean'].strip() != row['cleaned'].strip()
        if changed:
            print(f"    (LLM edited the text)")
    
    print(f"\n{'='*60}")
    print("SAMPLE LLM DROPPED:")
    print(f"{'='*60}")
    dropped_samples = llm_processed[~llm_processed['keep']].head(2)
    for idx, row in dropped_samples.iterrows():
        print(f"\n❌ [{idx}] Confidence: {row['confidence']:.2f}")
        print(f"    Content: {row['chunk_clean'][:150]}...")

def check_final():
    """Check final output."""
    parquet_path = "simple_wiki_clean.parquet"
    jsonl_path = "simple_wiki_clean.jsonl"
    
    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run: python clean_simple_wiki.py --stage finalize --test")
        return
    
    df = pd.read_parquet(parquet_path)
    print(f"\n{'='*60}")
    print(f"FINAL OUTPUT: {parquet_path}")
    print(f"{'='*60}")
    print(f"Total records kept: {len(df):,}")
    
    if 'decision_source' in df.columns:
        print(f"\nDecision source breakdown:")
        print(df['decision_source'].value_counts().to_string())
    
    print(f"\n{'='*60}")
    print("SAMPLE FINAL SENTENCES:")
    print(f"{'='*60}")
    for idx, row in df.head(5).iterrows():
        title_str = row['title'][:50] if row['title'] else '(no title)'
        sent_idx = f" [{row['sentence_idx']}]" if 'sentence_idx' in row else ''
        print(f"\n{sent_idx} From: {title_str}")
        print(f"    Decision: {row.get('decision_source', 'N/A')}")
        sentence = row.get('sentence', row.get('chunk_final', ''))
        print(f"    {sentence[:150]}...")
    
    if os.path.exists(jsonl_path):
        size_mb = os.path.getsize(jsonl_path) / (1024*1024)
        print(f"\n📄 JSONL file: {jsonl_path} ({size_mb:.2f} MB)")
        print(f"   Format: One sentence per line, ready for semantic parsing")

def main():
    stages = {
        'heuristics': check_heuristics,
        'classifier': check_classifier,
        'llm': check_llm,
        'final': check_final,
    }
    
    if len(sys.argv) > 1:
        stage = sys.argv[1].lower()
        if stage not in stages:
            print(f"Unknown stage: {stage}")
            print(f"Available stages: {', '.join(stages.keys())}")
            sys.exit(1)
        stages[stage]()
    else:
        # Auto-detect latest stage
        for stage_name in ['final', 'llm', 'classifier', 'heuristics']:
            if stage_name == 'final':
                path = 'simple_wiki_clean.parquet'
            else:
                path = f'simple_wiki_clean_{stage_name}.parquet'
            
            if os.path.exists(path):
                print(f"Auto-detected latest stage: {stage_name}")
                stages[stage_name]()
                break
        else:
            print("❌ No output files found. Run the pipeline first:")
            print("   python clean_simple_wiki.py --stage heuristics --test")

if __name__ == "__main__":
    main()

