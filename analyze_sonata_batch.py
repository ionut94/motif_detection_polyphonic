#!/usr/bin/env python3
"""
Analyze and compare batch sonata results with varying delta values (gamma=24).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_sonata_results(results_dir, delta_values, gamma=24):
    """Load all sonata summary CSV files for the given delta values."""
    all_results = []
    
    for delta in delta_values:
        file_path = os.path.join(results_dir, f'sonata_summary_per_sonata_{delta}_{gamma}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['delta'] = delta
            df['gamma'] = gamma
            all_results.append(df)
            print(f"Loaded: {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not all_results:
        return None
    
    return pd.concat(all_results, ignore_index=True)

def create_comparison_summary(df):
    """Create a summary table aggregated by delta."""
    summary = df.groupby(['delta', 'gamma']).agg({
        'total_occurrences': 'sum',
        'total_time_s': 'sum',
        'avg_time_s': 'mean',
        'avg_memory_mb': 'mean',
        'note_count': 'sum',
        'sonata': 'count'
    }).reset_index()
    
    summary.rename(columns={'sonata': 'num_sonatas'}, inplace=True)
    summary['avg_occurrences_per_sonata'] = summary['total_occurrences'] / summary['num_sonatas']
    
    return summary

def plot_analysis(df, output_dir):
    """Create visualization plots comparing metrics across delta values."""
    sns.set_style("whitegrid")
    
    # Create summary for plotting
    summary = create_comparison_summary(df)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Sonata Results: Impact of Delta (γ=24)', fontsize=16, fontweight='bold')
    
    # Plot 1: Total occurrences vs Delta
    ax1 = axes[0, 0]
    ax1.bar(summary['delta'], summary['total_occurrences'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Delta (δ)', fontsize=12)
    ax1.set_ylabel('Total Occurrences', fontsize=12)
    ax1.set_title('Total Motif Occurrences vs Delta', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(summary['delta'])
    
    # Add value labels on bars
    for i, row in summary.iterrows():
        ax1.text(row['delta'], row['total_occurrences'], f"{int(row['total_occurrences']):,}", 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average occurrences per sonata vs Delta
    ax2 = axes[0, 1]
    ax2.plot(summary['delta'], summary['avg_occurrences_per_sonata'], marker='o', 
            linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Delta (δ)', fontsize=12)
    ax2.set_ylabel('Avg Occurrences per Sonata', fontsize=12)
    ax2.set_title('Average Occurrences per Sonata vs Delta', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(summary['delta'])
    
    # Plot 3: Total execution time vs Delta
    ax3 = axes[1, 0]
    ax3.plot(summary['delta'], summary['total_time_s'], marker='s', 
            linewidth=2, markersize=8, color='darkorange')
    ax3.set_xlabel('Delta (δ)', fontsize=12)
    ax3.set_ylabel('Total Time (seconds)', fontsize=12)
    ax3.set_title('Total Execution Time vs Delta', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(summary['delta'])
    
    # Plot 4: Per-sonata box plot
    ax4 = axes[1, 1]
    
    # Prepare data for box plot
    box_data = [df[df['delta'] == d]['total_occurrences'].values for d in sorted(df['delta'].unique())]
    bp = ax4.boxplot(box_data, labels=sorted(df['delta'].unique()), patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Delta (δ)', fontsize=12)
    ax4.set_ylabel('Occurrences per Sonata', fontsize=12)
    ax4.set_title('Distribution of Occurrences per Sonata', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'sonata_batch_analysis_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()

def print_detailed_analysis(df, summary):
    """Print detailed analysis of the results."""
    print("\n" + "="*80)
    print("SONATA BATCH ANALYSIS: Delta Sensitivity (γ=24)")
    print("="*80)
    
    print("\n" + "-"*80)
    print("SUMMARY TABLE (Aggregated by Delta)")
    print("-"*80)
    print(summary.to_string(index=False))
    
    # Find best configurations
    print("\n" + "-"*80)
    print("BEST CONFIGURATIONS")
    print("-"*80)
    
    best_total_idx = summary['total_occurrences'].idxmax()
    fastest_idx = summary['total_time_s'].idxmin()
    
    print(f"\nMost Occurrences:  δ={summary.loc[best_total_idx, 'delta']} (Total: {summary.loc[best_total_idx, 'total_occurrences']:,})")
    print(f"Fastest Execution: δ={summary.loc[fastest_idx, 'delta']} (Time: {summary.loc[fastest_idx, 'total_time_s']:.2f}s)")
    
    # Analyze trends
    print("\n" + "-"*80)
    print("TREND ANALYSIS")
    print("-"*80)
    
    delta_min = summary['delta'].min()
    delta_max = summary['delta'].max()
    
    occ_start = summary[summary['delta'] == delta_min]['total_occurrences'].values[0]
    occ_end = summary[summary['delta'] == delta_max]['total_occurrences'].values[0]
    occ_change = occ_end - occ_start
    occ_pct = (occ_change / occ_start) * 100
    
    time_start = summary[summary['delta'] == delta_min]['total_time_s'].values[0]
    time_end = summary[summary['delta'] == delta_max]['total_time_s'].values[0]
    time_change = time_end - time_start
    time_pct = (time_change / time_start) * 100
    
    print(f"\nAs delta increases from {delta_min} to {delta_max}:")
    print(f"  - Total Occurrences: {occ_start:,.0f} → {occ_end:,.0f} ({occ_change:+,.0f}, {occ_pct:+.1f}%)")
    print(f"  - Total Time:        {time_start:.2f}s → {time_end:.2f}s ({time_change:+.2f}s, {time_pct:+.1f}%)")
    print(f"  - Avg Time/Sonata:   {summary[summary['delta'] == delta_min]['avg_time_s'].values[0]:.4f}s → {summary[summary['delta'] == delta_max]['avg_time_s'].values[0]:.4f}s")
    
    # Top sonatas by occurrences for each delta
    print("\n" + "-"*80)
    print("TOP 5 SONATAS BY OCCURRENCES (for each Delta)")
    print("-"*80)
    
    for delta in sorted(df['delta'].unique()):
        df_delta = df[df['delta'] == delta].nlargest(5, 'total_occurrences')
        print(f"\nδ={delta}:")
        for _, row in df_delta.iterrows():
            print(f"  Sonata {int(row['sonata']):2d}: {int(row['total_occurrences']):6,} occurrences (time: {row['total_time_s']:.3f}s)")
    
    print("\n" + "="*80)

def main():
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, 'results')
    
    delta_values = [2, 4, 8, 16]
    gamma = 24
    
    print(f"Loading sonata results for δ ∈ {delta_values}, γ={gamma}")
    print("-" * 80)
    
    # Load all results
    df = load_sonata_results(results_dir, delta_values, gamma)
    
    if df is None or df.empty:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(df)} sonata results across {len(delta_values)} delta values")
    
    # Create summary
    summary = create_comparison_summary(df)
    
    # Print detailed analysis
    print_detailed_analysis(df, summary)
    
    # Save summary table to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f'sonata_batch_comparison_{timestamp}.csv')
    summary.to_csv(csv_path, index=False)
    print(f"\nComparison summary saved to: {csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_analysis(df, results_dir)

if __name__ == "__main__":
    main()
