#!/usr/bin/env python3
"""
Quick utility to view rejected/dropped sentences from pipeline logs.
Usage: 
  python view_rejections.py [stage] [--reason REASON]
  
Stages: heuristics (default), classifier, llm, all
"""

import json
import sys
from collections import Counter
from pathlib import Path

def view_heuristics():
    """View heuristics stage rejections."""
    log_path = "logs/01_heuristics.jsonl"
    
    if not Path(log_path).exists():
        print(f"❌ {log_path} not found.")
        print("Run: python clean_simple_wiki.py --stage heuristics --test")
        return False
    
    # Load logs
    with open(log_path) as f:
        logs = [json.loads(line) for line in f]
    
    rejected = [l for l in logs if not l.get('accepted')]
    accepted = [l for l in logs if l.get('accepted')]
    
    # Filter by reason if requested
    filter_reason = None
    if '--reason' in sys.argv:
        idx = sys.argv.index('--reason')
        if idx + 1 < len(sys.argv):
            filter_reason = sys.argv[idx + 1]
            rejected = [r for r in rejected if r.get('filter_reason') == filter_reason]
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 HEURISTICS STAGE - Sentence Extraction & Filtering")
    print(f"{'='*60}")
    print(f"Total candidates: {len(logs)}")
    print(f"✅ Accepted: {len(accepted)} ({len(accepted)/len(logs)*100:.1f}%)")
    print(f"❌ Rejected: {len([l for l in logs if not l.get('accepted')])} ({len([l for l in logs if not l.get('accepted')])/len(logs)*100:.1f}%)")
    
    # Rejection reasons
    all_rejected = [l for l in logs if not l.get('accepted')]
    reasons = Counter(r.get('filter_reason') for r in all_rejected)
    print(f"\n📋 Rejection Reasons:")
    for reason, count in reasons.most_common():
        marker = " ← viewing" if reason == filter_reason else ""
        print(f"  • {reason}: {count} ({count/len(all_rejected)*100:.1f}%){marker}")
    
    # Show rejected sentences
    if filter_reason:
        print(f"\n{'='*60}")
        print(f"Showing rejected sentences with reason: {filter_reason}")
    else:
        print(f"\n{'='*60}")
        print(f"Showing all rejected sentences")
    
    print(f"{'='*60}\n")
    
    for i, r in enumerate(rejected, 1):
        reason = r.get('filter_reason', 'unknown')
        text = r.get('sentence_after') or r.get('fragment_before', '')
        title = r.get('title', '(no title)')[:60]
        source_idx = r.get('source_idx', -1)
        
        print(f"{i}. [{reason}] FROM: {title}")
        print(f"   Source idx: {source_idx}")
        print(f"   Text: {text[:200]}")
        if len(text) > 200:
            print(f"   ... (truncated, {len(text)} chars total)")
        print()
    
    if not rejected:
        print("No rejected sentences found.")
    else:
        if not filter_reason:
            print(f"\n💡 Tip: Filter by reason with: python view_rejections.py heuristics --reason <reason>")
            print(f"   Available reasons: {', '.join(reasons.keys())}")
    
    return True

def view_classifier():
    """View classifier stage decisions (samples only)."""
    log_path = "logs/02_classifier.jsonl"
    
    if not Path(log_path).exists():
        print(f"❌ {log_path} not found.")
        print("Run: python clean_simple_wiki.py --stage classifier --use_classifier --test")
        return False
    
    with open(log_path) as f:
        logs = [json.loads(line) for line in f]
    
    print(f"\n{'='*60}")
    print(f"📊 CLASSIFIER STAGE - Quality Scoring (Sample Log)")
    print(f"{'='*60}")
    print(f"Note: Classifier logs only contain samples, not all decisions.")
    print(f"Sample entries: {len(logs)}\n")
    
    # Group by bucket
    buckets = Counter(l.get('bucket') for l in logs)
    for bucket, count in buckets.most_common():
        print(f"  {bucket}: {count} samples")
    
    print(f"\n{'='*60}")
    print(f"Classifier Drop Examples (low quality, score ≤ 0.35)")
    print(f"{'='*60}\n")
    
    dropped = [l for l in logs if l.get('bucket') == 'classifier_drop']
    if not dropped:
        print("No drop examples in sample log.\n")
    else:
        for i, r in enumerate(dropped, 1):
            score = r.get('score', 0)
            text = r.get('sentence', '')
            title = r.get('title', '(no title)')[:60]
            
            print(f"{i}. [score: {score:.3f}] FROM: {title}")
            print(f"   Text: {text[:200]}")
            if len(text) > 200:
                print(f"   ... (truncated)")
            print()
    
    print(f"{'='*60}")
    print(f"Classifier Gray Zone Examples (dubious, 0.35-0.75 → sent to LLM)")
    print(f"{'='*60}\n")
    
    gray = [l for l in logs if l.get('bucket') == 'classifier_gray']
    if not gray:
        print("No gray zone examples in sample log.\n")
    else:
        for i, r in enumerate(gray, 1):
            score = r.get('score', 0)
            text = r.get('sentence', '')
            title = r.get('title', '(no title)')[:60]
            
            print(f"{i}. [score: {score:.3f}] FROM: {title}")
            print(f"   Text: {text[:200]}")
            if len(text) > 200:
                print(f"   ... (truncated)")
            print()
    
    return True

def view_llm():
    """View LLM stage rejections."""
    log_path = "logs/03_llm.jsonl"
    
    if not Path(log_path).exists():
        print(f"❌ {log_path} not found.")
        print("Run: python clean_simple_wiki.py --stage llm --use_llm --test")
        return False
    
    with open(log_path) as f:
        logs = [json.loads(line) for line in f]
    
    kept = [l for l in logs if l.get('keep', False)]
    dropped = [l for l in logs if not l.get('keep', False)]
    
    print(f"\n{'='*60}")
    print(f"📊 LLM STAGE - Human-like Quality Judgment")
    print(f"{'='*60}")
    print(f"Total reviewed: {len(logs)}")
    print(f"✅ Kept: {len(kept)} ({len(kept)/len(logs)*100:.1f}%)")
    print(f"❌ Dropped: {len(dropped)} ({len(dropped)/len(logs)*100:.1f}%)")
    
    if kept:
        avg_conf_kept = sum(l.get('confidence', 0) for l in kept) / len(kept)
        print(f"Avg confidence (kept): {avg_conf_kept:.3f}")
    if dropped:
        avg_conf_dropped = sum(l.get('confidence', 0) for l in dropped) / len(dropped)
        print(f"Avg confidence (dropped): {avg_conf_dropped:.3f}")
    
    print(f"\n{'='*60}")
    print(f"LLM Dropped Sentences")
    print(f"{'='*60}\n")
    
    if not dropped:
        print("No dropped sentences.\n")
    else:
        for i, r in enumerate(dropped, 1):
            conf = r.get('confidence', 0)
            text = r.get('sentence_original', r.get('sentence', ''))
            title = r.get('title', '(no title)')[:60]
            context = r.get('context', '')
            
            print(f"{i}. [confidence: {conf:.3f}] FROM: {title}")
            if context:
                print(f"   Context: {context[:150]}")
            print(f"   Dropped: {text[:200]}")
            if len(text) > 200:
                print(f"   ... (truncated)")
            print()
    
    print(f"\n{'='*60}")
    print(f"LLM Kept & Rewritten (showing rewrite examples)")
    print(f"{'='*60}\n")
    
    rewritten = [l for l in kept if l.get('sentence_original') != l.get('sentence_cleaned')][:5]
    if not rewritten:
        print("No rewrite examples (or not logged).\n")
    else:
        for i, r in enumerate(rewritten, 1):
            conf = r.get('confidence', 0)
            original = r.get('sentence_original', '')
            cleaned = r.get('sentence_cleaned', '')
            title = r.get('title', '(no title)')[:60]
            
            print(f"{i}. [confidence: {conf:.3f}] FROM: {title}")
            print(f"   Original: {original[:150]}")
            print(f"   Rewritten: {cleaned[:150]}")
            print()
    
    return True

def main():
    # Determine stage
    stage = 'heuristics'  # default
    if len(sys.argv) > 1 and sys.argv[1] in ['heuristics', 'classifier', 'llm', 'all']:
        stage = sys.argv[1]
    
    success = False
    
    if stage == 'all':
        print("\n" + "="*60)
        print("VIEWING ALL PIPELINE REJECTIONS")
        print("="*60)
        view_heuristics()
        view_classifier()
        view_llm()
        return
    elif stage == 'heuristics':
        success = view_heuristics()
    elif stage == 'classifier':
        success = view_classifier()
    elif stage == 'llm':
        success = view_llm()
    
    if not success:
        print("\n💡 Usage: python view_rejections.py [stage] [--reason REASON]")
        print("   Stages: heuristics (default), classifier, llm, all")
        print("   Example: python view_rejections.py heuristics --reason length")
        print("   Example: python view_rejections.py llm")
        print("   Example: python view_rejections.py all")

if __name__ == "__main__":
    main()
