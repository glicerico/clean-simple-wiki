#!/usr/bin/env python3
"""
Generate HTML report for reviewing pipeline outputs.
Usage: python generate_report.py [stage_name]
  stage_name: heuristics, classifier, llm, or final (default: heuristics)
"""

import sys
import os
import json
import pandas as pd
from collections import Counter
from pathlib import Path

def generate_heuristics_report():
    """Generate interactive HTML report for heuristics stage."""
    parquet_path = "simple_wiki_clean_heuristics.parquet"
    log_path = "logs/01_heuristics.jsonl"
    output_path = "report_heuristics.html"
    
    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run: python clean_simple_wiki.py --stage heuristics --test")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    # Calculate statistics from parquet (these are the ACCEPTED sentences)
    num_accepted = len(df)
    num_sources = df["source_idx"].nunique() if "source_idx" in df.columns else 0
    avg_per_source = num_accepted / num_sources if num_sources > 0 else 0
    
    # Try to get total candidates from source metadata
    if "source_num_sentences_total" in df.columns:
        # This is an estimate based on source documents
        total_from_sources = df.groupby("source_idx")["source_num_sentences_total"].first().sum()
    else:
        total_from_sources = None
    
    # Load log data for detailed rejection info
    logs = []
    logs_available = False
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
            logs_available = True
        except:
            pass
    
    # Calculate processing stats from logs if available
    if logs_available and logs:
        rejected_logs = [l for l in logs if not l.get('accepted', False)]
        accepted_logs = [l for l in logs if l.get('accepted', False)]
        reason_counts = Counter(r.get('filter_reason', 'unknown') for r in rejected_logs if r.get('filter_reason'))
        
        total_candidates = len(logs)
        num_rejected = len(rejected_logs)
        acceptance_rate = (len(accepted_logs) / total_candidates * 100) if total_candidates > 0 else 0
        rejection_rate = (num_rejected / total_candidates * 100) if total_candidates > 0 else 0
    else:
        # No logs - use estimates from parquet
        reason_counts = Counter()
        if total_from_sources:
            total_candidates = int(total_from_sources)
            num_rejected = total_candidates - num_accepted
            acceptance_rate = (num_accepted / total_candidates * 100) if total_candidates > 0 else 0
            rejection_rate = (num_rejected / total_candidates * 100) if total_candidates > 0 else 0
        else:
            total_candidates = num_accepted
            num_rejected = 0
            acceptance_rate = 100.0
            rejection_rate = 0.0
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Heuristics Stage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-card.warning {{
            border-left-color: #ff9800;
        }}
        .stat-card.info {{
            border-left-color: #2196F3;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .reason-chart {{
            margin: 20px 0;
        }}
        .reason-bar {{
            margin: 10px 0;
        }}
        .reason-label {{
            font-weight: 500;
            margin-bottom: 4px;
        }}
        .bar-container {{
            background: #e0e0e0;
            height: 25px;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            background: linear-gradient(90deg, #ff5252, #ff1744);
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 8px;
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .record {{
            margin: 20px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .record-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .record-title {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
            flex: 1;
        }}
        .record-stats {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #666;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge.success {{
            background: #c8e6c9;
            color: #2e7d32;
        }}
        .badge.warning {{
            background: #ffecb3;
            color: #f57c00;
        }}
        .text-box {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 4px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }}
        .original {{
            background: #ffebee;
            border-left: 4px solid #e53935;
        }}
        .cleaned {{
            background: #e8f5e9;
            border-left: 4px solid #43a047;
        }}
        .empty {{
            background: #fafafa;
            border-left: 4px solid #999;
            color: #999;
        }}
        .section-label {{
            font-size: 0.85em;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .sentence-log {{
            margin: 15px 0;
            padding: 12px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #666;
        }}
        .sentence-log.accepted {{
            border-left-color: #4CAF50;
        }}
        .sentence-log.rejected {{
            border-left-color: #f44336;
        }}
        .log-reason {{
            display: inline-block;
            padding: 2px 8px;
            background: #f44336;
            color: white;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 8px;
        }}
        .filter-controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 6px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            background: #2196F3;
            color: white;
        }}
        .filter-btn:hover {{
            background: #1976D2;
        }}
        .filter-btn.active {{
            background: #4CAF50;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Heuristics Stage Report</h1>
        <p>Review of sentence extraction and filtering decisions.</p>
        
        <h2>📈 Overall Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Source Documents</div>
                <div class="stat-value">{num_sources:,}</div>
            </div>
            <div class="stat-card info">
                <div class="stat-label">Total Candidates</div>
                <div class="stat-value">{total_candidates:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">✅ Accepted</div>
                <div class="stat-value">{num_accepted:,}</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">❌ Rejected</div>
                <div class="stat-value">{num_rejected:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Acceptance Rate</div>
                <div class="stat-value">{acceptance_rate:.1f}%</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">Rejection Rate</div>
                <div class="stat-value">{rejection_rate:.1f}%</div>
            </div>
            <div class="stat-card info">
                <div class="stat-label">Avg Kept per Doc</div>
                <div class="stat-value">{avg_per_source:.1f}</div>
            </div>
        </div>
        <p style="color: #666; margin-top: 10px;">
            ℹ️ Each output record is one clean sentence, ready for semantic parsing
        </p>
"""
    
    if not logs_available:
        html += """
        <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 6px; margin: 20px 0;">
            ⚠️ <strong>Note:</strong> Detailed rejection statistics unavailable (logs not found). 
            Run the pipeline to generate logs for full breakdown.
        </div>
"""
    
    # Add rejection reasons chart with examples
    if reason_counts and logs_available:
        html += """
        <h2>❌ Rejection Reasons</h2>
        <div class="reason-chart">
"""
        total_rejected = sum(reason_counts.values())
        for reason, count in reason_counts.most_common():
            percentage = (count / total_rejected) * 100
            html += f"""
            <div class="reason-bar">
                <div class="reason-label">{reason}: {count:,} ({percentage:.1f}%)</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {percentage}%">{count:,}</div>
                </div>
            </div>
"""
        html += """
        </div>
        
        <h2>📋 Rejection Examples</h2>
        <p style="color: #666; margin-bottom: 15px;">Sample sentences that were filtered out, by rejection reason:</p>
"""
        
        # Group rejected logs by reason and show examples
        from collections import defaultdict
        examples_by_reason = defaultdict(list)
        for log in rejected_logs:
            reason = log.get('filter_reason', 'unknown')
            if reason and len(examples_by_reason[reason]) < 3:  # Keep up to 3 examples per reason
                text = log.get('sentence_after') if log.get('sentence_after') else log.get('fragment_before', '')
                if text:
                    examples_by_reason[reason].append(text)
        
        for reason, count in reason_counts.most_common():
            if reason in examples_by_reason:
                html += f"""
        <div class="record">
            <div class="record-header">
                <div class="record-title">❌ {reason.replace('_', ' ').title()}</div>
                <div class="record-stats">
                    <span class="badge warning">{count:,} rejected</span>
                </div>
            </div>
"""
                for i, example in enumerate(examples_by_reason[reason], 1):
                    html += f"""
            <div class="section-label">Example {i}</div>
            <div class="text-box original">{example[:200]}</div>
"""
                html += """
        </div>
"""
        html += """
"""
    
    # Add sample sentences
    html += """
        <h2>📝 Sample Sentences</h2>
        <div id="records">
"""
    
    # Add sample sentences
    num_samples = min(50, len(df))
    for idx in range(num_samples):
        row = df.iloc[idx]
        
        title_str = row['title'][:80] if row['title'] else '(no title)'
        sent_idx = row.get('sentence_idx', 0)
        sentence = row.get('sentence', '')
        
        html += f"""
        <div class="record">
            <div class="record-header">
                <div class="record-title">{title_str}</div>
                <div class="record-stats">
                    <span class="badge success">Sentence #{sent_idx}</span>
                </div>
            </div>
            
            <div class="text-box cleaned">{sentence}</div>
        </div>
"""
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")
    return output_path

def generate_final_report():
    """Generate report for final output."""
    parquet_path = "simple_wiki_clean.parquet"
    output_path = "report_final.html"
    
    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run: python clean_simple_wiki.py --stage finalize")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    total_sentences = len(df)
    num_sources = df["source_idx"].nunique() if "source_idx" in df.columns else 0
    avg_per_source = total_sentences / num_sources if num_sources > 0 else 0
    
    # Decision source breakdown
    decision_breakdown = {}
    if "decision_source" in df.columns:
        decision_breakdown = df["decision_source"].value_counts().to_dict()
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Final Output Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .record {{
            margin: 15px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .sentence {{ font-size: 1.05em; line-height: 1.6; color: #333; }}
        .meta {{ font-size: 0.85em; color: #666; margin-top: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✅ Final Output Report</h1>
        <p>Clean sentences ready for semantic parsing.</p>
        
        <h2>📈 Final Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Clean Sentences</div>
                <div class="stat-value">{total_sentences:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Source Documents</div>
                <div class="stat-value">{num_sources:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg per Document</div>
                <div class="stat-value">{avg_per_source:.1f}</div>
            </div>
        </div>
"""
    
    if decision_breakdown:
        html += """
        <h2>📊 Decision Source Breakdown</h2>
        <div style="margin: 20px 0;">
"""
        for source, count in sorted(decision_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_sentences * 100) if total_sentences > 0 else 0
            html += f"""
            <div style="margin: 10px 0;">
                <strong>{source}:</strong> {count:,} ({percentage:.1f}%)
            </div>
"""
        html += """
        </div>
"""
    
    html += """
        <h2>📝 Sample Clean Sentences (First 30)</h2>
        <div>
"""
    
    for idx in range(min(30, len(df))):
        row = df.iloc[idx]
        sentence = row.get('sentence', '')
        title = row.get('title', '(no title)')[:60]
        
        html += f"""
        <div class="record">
            <div class="sentence">{sentence}</div>
            <div class="meta">From: {title}</div>
        </div>
"""
    
    html += """
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #e8f5e9; border-radius: 6px;">
            <h3 style="margin-top: 0; color: #2e7d32;">✅ Ready for Semantic Parsing</h3>
            <p style="margin: 0;">
                Output files: <code>simple_wiki_clean.jsonl</code> and <code>simple_wiki_clean.parquet</code><br>
                Each record contains one clean sentence with metadata (title, source_idx, split).
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")
    return output_path

def generate_classifier_report():
    """Generate report for classifier stage."""
    parquet_path = "simple_wiki_clean_classifier.parquet"
    log_path = "logs/02_classifier.jsonl"
    output_path = "report_classifier.html"
    
    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run: python clean_simple_wiki.py --stage classifier --use_classifier --test")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    total_sentences = len(df)
    
    # Decision breakdown
    decision_counts = df['decision_source'].value_counts().to_dict() if 'decision_source' in df.columns else {}
    
    # Score statistics
    if 'score_classifier' in df.columns:
        scores = df['score_classifier'].dropna()
        score_mean = scores.mean()
        score_median = scores.median()
        score_min = scores.min()
        score_max = scores.max()
    else:
        score_mean = score_median = score_min = score_max = 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Classifier Stage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #2196F3;
        }}
        .stat-card.success {{ border-left-color: #4CAF50; }}
        .stat-card.warning {{ border-left-color: #ff9800; }}
        .stat-card.danger {{ border-left-color: #f44336; }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .record {{
            margin: 15px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 6px;
            border-left: 4px solid #ddd;
        }}
        .record.keep {{ border-left-color: #4CAF50; }}
        .record.drop {{ border-left-color: #f44336; }}
        .record.gray {{ border-left-color: #ff9800; }}
        .sentence {{ font-size: 1.05em; line-height: 1.6; color: #333; margin: 10px 0; }}
        .meta {{ font-size: 0.85em; color: #666; margin-top: 8px; }}
        .score {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
        .score.high {{ background: #c8e6c9; color: #2e7d32; }}
        .score.low {{ background: #ffcdd2; color: #c62828; }}
        .score.mid {{ background: #ffe0b2; color: #e65100; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Classifier Stage Report</h1>
        <p>Quality scoring and sentence triage using zero-shot classification.</p>
        
        <h2>📈 Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Sentences</div>
                <div class="stat-value">{total_sentences:,}</div>
            </div>
            <div class="stat-card success">
                <div class="stat-label">✅ Auto-Keep</div>
                <div class="stat-value">{decision_counts.get('classifier_keep', 0):,}</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-label">❌ Auto-Drop</div>
                <div class="stat-value">{decision_counts.get('classifier_drop', 0):,}</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">⚠️ Gray Zone (→ LLM)</div>
                <div class="stat-value">{decision_counts.get('classifier_gray', 0):,}</div>
            </div>
        </div>
        
        <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
            <strong>Score Statistics:</strong><br>
            Mean: {score_mean:.3f} | Median: {score_median:.3f} | Range: {score_min:.3f} - {score_max:.3f}
        </div>
        
        <div style="background: #fff3e0; padding: 15px; border-radius: 6px; margin: 20px 0;">
            <strong>Thresholds:</strong><br>
            ≥ 0.75: Auto-KEEP (high quality)<br>
            ≤ 0.35: Auto-DROP (low quality)<br>
            0.35 - 0.75: Gray zone → needs LLM review
        </div>
"""
    
    # Show examples from each category
    for decision, label, color in [
        ('classifier_keep', '✅ High Quality (Auto-Keep)', 'keep'),
        ('classifier_drop', '❌ Low Quality (Auto-Drop)', 'drop'),
        ('classifier_gray', '⚠️ Gray Zone (Needs LLM)', 'gray')
    ]:
        samples = df[df['decision_source'] == decision].head(5) if decision in df['decision_source'].values else pd.DataFrame()
        
        if len(samples) > 0:
            html += f"""
        <h2>{label}</h2>
        <div>
"""
            for idx, row in samples.iterrows():
                score = row.get('score_classifier', 0)
                score_class = 'high' if score >= 0.75 else ('low' if score <= 0.35 else 'mid')
                sentence = row.get('sentence', '')
                title = row.get('title', '(no title)')[:50]
                
                html += f"""
            <div class="record {color}">
                <div class="meta">
                    From: {title} | 
                    Score: <span class="score {score_class}">{score:.3f}</span>
                </div>
                <div class="sentence">{sentence}</div>
            </div>
"""
            html += """
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")
    return output_path

def generate_llm_report():
    """Generate report for LLM stage."""
    parquet_path = "simple_wiki_clean_llm.parquet"
    log_path = "logs/03_llm.jsonl"
    output_path = "report_llm.html"
    
    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} not found. Run: python clean_simple_wiki.py --stage llm --use_llm --test")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    # Load logs if available for more detailed examples
    log_records = []
    if os.path.exists(log_path):
        log_records = read_jsonl(log_path)
    
    total_sentences = len(df)
    
    # Decision breakdown
    decision_counts = df['decision_source'].value_counts().to_dict() if 'decision_source' in df.columns else {}
    
    llm_processed = df[df['decision_source'] == 'llm'] if 'decision_source' in df.columns else pd.DataFrame()
    num_llm_keep = llm_processed[llm_processed['keep']].shape[0] if len(llm_processed) > 0 else 0
    num_llm_drop = llm_processed[~llm_processed['keep']].shape[0] if len(llm_processed) > 0 else 0
    
    # Confidence statistics
    if len(llm_processed) > 0 and 'confidence' in llm_processed.columns:
        conf_mean = llm_processed['confidence'].mean()
        conf_median = llm_processed['confidence'].median()
    else:
        conf_mean = conf_median = 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Stage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 3px solid #9c27b0; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #9c27b0;
        }}
        .stat-card.success {{ border-left-color: #4CAF50; }}
        .stat-card.danger {{ border-left-color: #f44336; }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .record {{
            margin: 15px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 6px;
            border-left: 4px solid #ddd;
        }}
        .record.keep {{ border-left-color: #4CAF50; }}
        .record.drop {{ border-left-color: #f44336; }}
        .sentence {{ font-size: 1.05em; line-height: 1.6; color: #333; margin: 10px 0; }}
        .meta {{ font-size: 0.85em; color: #666; margin-top: 8px; }}
        .confidence {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
        .confidence.high {{ background: #c8e6c9; color: #2e7d32; }}
        .confidence.low {{ background: #ffcdd2; color: #c62828; }}
        .confidence.mid {{ background: #ffe0b2; color: #e65100; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 LLM Stage Report</h1>
        <p>GPT-4o-mini review of borderline sentences (gray zone from classifier).</p>
        
        <h2>📈 Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Sentences</div>
                <div class="stat-value">{total_sentences:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">LLM Processed</div>
                <div class="stat-value">{len(llm_processed):,}</div>
            </div>
            <div class="stat-card success">
                <div class="stat-label">✅ LLM Kept</div>
                <div class="stat-value">{num_llm_keep:,}</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-label">❌ LLM Dropped</div>
                <div class="stat-value">{num_llm_drop:,}</div>
            </div>
        </div>
        
        <div style="background: #f3e5f5; padding: 15px; border-radius: 6px; margin: 20px 0;">
            <strong>LLM Confidence:</strong><br>
            Mean: {conf_mean:.3f} | Median: {conf_median:.3f}
        </div>
        
        <div style="background: #e8f5e9; padding: 15px; border-radius: 6px; margin: 20px 0;">
            <strong>Note:</strong> LLM only reviews sentences that scored 0.35-0.75 in the classifier stage.
            High-quality (≥0.75) and low-quality (≤0.35) sentences were already decided.
        </div>
"""
    
    # Show examples from logs (more detailed) or fallback to parquet
    if log_records:
        kept_logs = [r for r in log_records if r.get('keep', False)][:5]
        dropped_logs = [r for r in log_records if not r.get('keep', True)][:5]
        
        if kept_logs:
            html += """
        <h2>✅ LLM Kept (with Context & Rewriting)</h2>
        <div>
"""
            for rec in kept_logs:
                conf = rec.get('confidence', 0)
                conf_class = 'high' if conf >= 0.7 else ('low' if conf <= 0.3 else 'mid')
                title = rec.get('title', '(no title)')[:50]
                context = rec.get('context', '')
                original = rec.get('sentence_original', '')
                cleaned = rec.get('sentence_cleaned', original)
                
                html += f"""
            <div class="record keep">
                <div class="meta">
                    From: {title} | 
                    Confidence: <span class="confidence {conf_class}">{conf:.3f}</span>
                </div>
"""
                if context:
                    html += f"""
                <div style="background: #e8f5e9; padding: 8px; margin: 8px 0; border-radius: 4px; font-size: 0.9em;">
                    <strong>Context:</strong> {context}
                </div>
"""
                if original != cleaned:
                    html += f"""
                <div style="background: #ffebee; padding: 8px; margin: 8px 0; border-radius: 4px;">
                    <strong>Original:</strong> {original}
                </div>
                <div style="background: #c8e6c9; padding: 8px; margin: 8px 0; border-radius: 4px;">
                    <strong>Cleaned:</strong> {cleaned}
                </div>
"""
                else:
                    html += f"""
                <div class="sentence">{cleaned}</div>
"""
                html += """
            </div>
"""
            html += """
        </div>
"""
        
        if dropped_logs:
            html += """
        <h2>❌ LLM Dropped</h2>
        <div>
"""
            for rec in dropped_logs:
                conf = rec.get('confidence', 0)
                conf_class = 'high' if conf >= 0.7 else ('low' if conf <= 0.3 else 'mid')
                title = rec.get('title', '(no title)')[:50]
                context = rec.get('context', '')
                original = rec.get('sentence_original', '')
                
                html += f"""
            <div class="record drop">
                <div class="meta">
                    From: {title} | 
                    Confidence: <span class="confidence {conf_class}">{conf:.3f}</span>
                </div>
"""
                if context:
                    html += f"""
                <div style="background: #fff3e0; padding: 8px; margin: 8px 0; border-radius: 4px; font-size: 0.9em;">
                    <strong>Context:</strong> {context}
                </div>
"""
                html += f"""
                <div class="sentence">{original}</div>
            </div>
"""
            html += """
        </div>
"""
    elif len(llm_processed) > 0:
        # Fallback to parquet if no logs
        kept = llm_processed[llm_processed['keep']].head(5)
        if len(kept) > 0:
            html += """
        <h2>✅ LLM Kept</h2>
        <div>
"""
            for idx, row in kept.iterrows():
                conf = row.get('confidence', 0)
                conf_class = 'high' if conf >= 0.7 else ('low' if conf <= 0.3 else 'mid')
                sentence = row.get('sentence', '')
                title = row.get('title', '(no title)')[:50]
                
                html += f"""
            <div class="record keep">
                <div class="meta">
                    From: {title} | 
                    Confidence: <span class="confidence {conf_class}">{conf:.3f}</span>
                </div>
                <div class="sentence">{sentence}</div>
            </div>
"""
            html += """
        </div>
"""
        
        dropped = llm_processed[~llm_processed['keep']].head(5)
        if len(dropped) > 0:
            html += """
        <h2>❌ LLM Dropped</h2>
        <div>
"""
            for idx, row in dropped.iterrows():
                conf = row.get('confidence', 0)
                conf_class = 'high' if conf >= 0.7 else ('low' if conf <= 0.3 else 'mid')
                sentence = row.get('sentence', '')
                title = row.get('title', '(no title)')[:50]
                
                html += f"""
            <div class="record drop">
                <div class="meta">
                    From: {title} | 
                    Confidence: <span class="confidence {conf_class}">{conf:.3f}</span>
                </div>
                <div class="sentence">{sentence}</div>
            </div>
"""
            html += """
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")
    return output_path

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not stage:
        # Auto-detect
        if os.path.exists('simple_wiki_clean.parquet'):
            print("Auto-detected: final stage")
            generate_final_report()
        elif os.path.exists('simple_wiki_clean_llm.parquet'):
            print("Auto-detected: llm stage")
            generate_llm_report()
        elif os.path.exists('simple_wiki_clean_classifier.parquet'):
            print("Auto-detected: classifier stage")
            generate_classifier_report()
        elif os.path.exists('simple_wiki_clean_heuristics.parquet'):
            print("Auto-detected: heuristics stage")
            generate_heuristics_report()
        else:
            print("❌ No output files found. Run the pipeline first.")
    elif stage == 'heuristics':
        generate_heuristics_report()
    elif stage == 'classifier':
        generate_classifier_report()
    elif stage == 'llm':
        generate_llm_report()
    elif stage in ['final', 'finalize']:
        generate_final_report()
    else:
        print(f"Unknown stage: {stage}")
        print("Available stages: heuristics, classifier, llm, final")

if __name__ == "__main__":
    main()

