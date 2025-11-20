#!/usr/bin/env python3
"""
Analyze and compare batch benchmark results with varying delta values.
"""

import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_benchmark_results(results_dir, timestamp_pattern="20251117_1535*"):
    """Load all benchmark JSON files matching the timestamp pattern."""
    pattern = os.path.join(results_dir, f"benchmark_beethoven_{timestamp_pattern}.json")
    files = sorted(glob.glob(pattern))
    
    # Also check the 1534 timestamps
    pattern2 = os.path.join(results_dir, "benchmark_beethoven_20251117_1534*.json")
    files.extend(sorted(glob.glob(pattern2)))
    files = sorted(list(set(files)))  # Remove duplicates and sort
    
    results = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def extract_summary_metrics(benchmark_data):
    """Extract key metrics from a benchmark result."""
    # Get first result to extract delta and gamma
    first_result = benchmark_data['results'][0]
    delta = first_result['delta']
    gamma = first_result['gamma']
    
    # Calculate micro-averaged metrics
    total_tp = sum(r['true_positives'] for r in benchmark_data['results'])
    total_fp = sum(r['false_positives'] for r in benchmark_data['results'])
    total_fn = sum(r['false_negatives'] for r in benchmark_data['results'])
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate macro-averaged metrics
    macro_precision = sum(r['precision'] for r in benchmark_data['results']) / len(benchmark_data['results'])
    macro_recall = sum(r['recall'] for r in benchmark_data['results']) / len(benchmark_data['results'])
    macro_f1 = sum(r['f1_score'] for r in benchmark_data['results']) / len(benchmark_data['results'])
    
    return {
        'delta': delta,
        'gamma': gamma,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_execution_time': sum(r['execution_time'] for r in benchmark_data['results']),
        'avg_execution_time': sum(r['execution_time'] for r in benchmark_data['results']) / len(benchmark_data['results'])
    }

def create_comparison_table(all_results):
    """Create a pandas DataFrame comparing all results."""
    summary_data = []
    
    for result in all_results:
        metrics = extract_summary_metrics(result)
        summary_data.append(metrics)
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('delta')
    
    return df

def plot_metrics_comparison(df, output_dir):
    """Create visualization plots comparing metrics across delta values."""
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beethoven Benchmark: Impact of Delta (γ=24)', fontsize=16, fontweight='bold')
    
    # Plot 1: Micro-averaged metrics
    ax1 = axes[0, 0]
    ax1.plot(df['delta'], df['micro_precision'], marker='o', label='Precision', linewidth=2)
    ax1.plot(df['delta'], df['micro_recall'], marker='s', label='Recall', linewidth=2)
    ax1.plot(df['delta'], df['micro_f1'], marker='^', label='F1-Score', linewidth=2)
    ax1.set_xlabel('Delta (δ)', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Micro-Averaged Metrics vs Delta', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['delta'])
    
    # Plot 2: Macro-averaged metrics
    ax2 = axes[0, 1]
    ax2.plot(df['delta'], df['macro_precision'], marker='o', label='Precision', linewidth=2)
    ax2.plot(df['delta'], df['macro_recall'], marker='s', label='Recall', linewidth=2)
    ax2.plot(df['delta'], df['macro_f1'], marker='^', label='F1-Score', linewidth=2)
    ax2.set_xlabel('Delta (δ)', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Macro-Averaged Metrics vs Delta', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['delta'])
    
    # Plot 3: True Positives vs False Positives
    ax3 = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df['total_tp'], width, label='True Positives', color='green', alpha=0.7)
    ax3.bar([i + width/2 for i in x], df['total_fp']/1000, width, label='False Positives (×1000)', color='red', alpha=0.7)
    ax3.set_xlabel('Delta (δ)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('True Positives vs False Positives', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['delta'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Execution Time
    ax4 = axes[1, 1]
    ax4.bar(range(len(df)), df['total_execution_time'], color='steelblue', alpha=0.7)
    ax4.set_xlabel('Delta (δ)', fontsize=12)
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.set_title('Total Execution Time vs Delta', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['delta'])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'batch_analysis_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()

def print_detailed_analysis(df):
    """Print detailed analysis of the results."""
    print("\n" + "="*80)
    print("BATCH BENCHMARK ANALYSIS: Delta Sensitivity (γ=24)")
    print("="*80)
    
    print("\n" + "-"*80)
    print("SUMMARY TABLE (Micro-Averaged Metrics)")
    print("-"*80)
    print(df[['delta', 'total_tp', 'total_fp', 'total_fn', 'micro_precision', 'micro_recall', 'micro_f1']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("SUMMARY TABLE (Macro-Averaged Metrics)")
    print("-"*80)
    print(df[['delta', 'macro_precision', 'macro_recall', 'macro_f1', 'total_execution_time']].to_string(index=False))
    
    # Find best configurations
    print("\n" + "-"*80)
    print("BEST CONFIGURATIONS")
    print("-"*80)
    
    best_micro_f1_idx = df['micro_f1'].idxmax()
    best_micro_recall_idx = df['micro_recall'].idxmax()
    best_micro_precision_idx = df['micro_precision'].idxmax()
    
    print(f"\nBest Micro F1-Score: δ={df.loc[best_micro_f1_idx, 'delta']} (F1={df.loc[best_micro_f1_idx, 'micro_f1']:.4f})")
    print(f"Best Micro Recall:   δ={df.loc[best_micro_recall_idx, 'delta']} (Recall={df.loc[best_micro_recall_idx, 'micro_recall']:.4f})")
    print(f"Best Micro Precision: δ={df.loc[best_micro_precision_idx, 'delta']} (Precision={df.loc[best_micro_precision_idx, 'micro_precision']:.4f})")
    
    # Analyze trends
    print("\n" + "-"*80)
    print("TREND ANALYSIS")
    print("-"*80)
    
    delta_change = df['delta'].iloc[-1] - df['delta'].iloc[0]
    tp_change = df['total_tp'].iloc[-1] - df['total_tp'].iloc[0]
    fp_change = df['total_fp'].iloc[-1] - df['total_fp'].iloc[0]
    recall_change = df['micro_recall'].iloc[-1] - df['micro_recall'].iloc[0]
    precision_change = df['micro_precision'].iloc[-1] - df['micro_precision'].iloc[0]
    
    print(f"\nAs delta increases from {df['delta'].iloc[0]} to {df['delta'].iloc[-1]}:")
    print(f"  - True Positives:  {df['total_tp'].iloc[0]} → {df['total_tp'].iloc[-1]} ({tp_change:+d}, {tp_change/df['total_tp'].iloc[0]*100:+.1f}%)")
    print(f"  - False Positives: {df['total_fp'].iloc[0]} → {df['total_fp'].iloc[-1]} ({fp_change:+d}, {fp_change/df['total_fp'].iloc[0]*100:+.1f}%)")
    print(f"  - Micro Recall:    {df['micro_recall'].iloc[0]:.4f} → {df['micro_recall'].iloc[-1]:.4f} ({recall_change:+.4f})")
    print(f"  - Micro Precision: {df['micro_precision'].iloc[0]:.4f} → {df['micro_precision'].iloc[-1]:.4f} ({precision_change:+.4f})")
    
    print("\n" + "="*80)

def main():
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, 'results')
    
    # Load all benchmark results from today's batch
    print("Loading benchmark results...")
    all_results = load_benchmark_results(results_dir)
    
    if not all_results:
        print("No benchmark results found!")
        return
    
    print(f"Found {len(all_results)} benchmark result files")
    
    # Create comparison dataframe
    df = create_comparison_table(all_results)
    
    # Print detailed analysis
    print_detailed_analysis(df)
    
    # Save comparison table to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f'batch_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(df, results_dir)

if __name__ == "__main__":
    main()
