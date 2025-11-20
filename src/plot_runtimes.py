import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = 'results'
BACKUP_DIR = os.path.join(RESULTS_DIR, 'backup_before_rerun')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(directory, label):
    data = []
    pattern = re.compile(r'sonata_summary_per_sonata_(\d+)_(\d+)\.csv')
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return pd.DataFrame()

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            delta = int(match.group(1))
            gamma = int(match.group(2))
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                df['delta'] = delta
                df['gamma'] = gamma
                df['version'] = label
                data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if not data:
        return pd.DataFrame()
    
    return pd.concat(data, ignore_index=True)

def plot_fixed_delta(df):
    # Filter for requested gamma values
    target_gammas = [2, 4, 8, 16, 24]
    df_filtered = df[df['gamma'].isin(target_gammas)].copy()
    
    # Convert gamma to categorical for consistent plotting
    df_filtered['gamma'] = df_filtered['gamma'].astype(str)
    gamma_order = [str(g) for g in target_gammas]
    
    # Define styles
    markers = ['o', 's', '^', 'D', 'X']
    dashes = [(1, 0), (4, 1.5), (1, 1), (3, 1, 1, 1), (5, 1, 1, 1, 1, 1)] # Solid, Dashed, Dotted, Dash-dot, Dense Dash-dot
    palette = sns.color_palette("bright", n_colors=len(target_gammas))

    deltas = sorted(df_filtered['delta'].unique())
    
    for delta in deltas:
        subset = df_filtered[df_filtered['delta'] == delta]
        if subset.empty:
            continue
        
        # Check if we have both versions for this delta
        versions = subset['version'].unique()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        
        # Determine common y-limit
        y_max = subset['total_time_s'].max() * 1.1
        
        for i, version in enumerate(['Old', 'New']):
            ax = axes[i]
            version_subset = subset[subset['version'] == version]
            
            if version_subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{version} Results')
                continue

            sns.lineplot(
                data=version_subset, 
                x='note_count', 
                y='total_time_s', 
                hue='gamma', 
                hue_order=gamma_order,
                style='gamma', 
                style_order=gamma_order,
                markers=markers, 
                dashes=dashes, 
                palette=palette,
                linewidth=2,
                markersize=8,
                ax=ax
            )
            
            ax.set_title(f'{version} Results (Fixed $\delta$={delta})')
            ax.set_xlabel('Number of Notes')
            if i == 0:
                ax.set_ylabel('Total Runtime (s)')
            else:
                ax.set_ylabel('')
            
            ax.set_ylim(bottom=0, top=y_max)
            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.3)
            ax.legend(title='$\gamma$')

        plt.tight_layout()
        output_path = os.path.join(PLOTS_DIR, f'runtime_comparison_fixed_delta_{delta}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_fixed_gamma(df):
    # Filter for requested delta values
    target_deltas = [2, 4, 8, 16, 24]
    df_filtered = df[df['delta'].isin(target_deltas)].copy()
    
    # Convert delta to categorical for consistent plotting
    df_filtered['delta'] = df_filtered['delta'].astype(str)
    delta_order = [str(d) for d in target_deltas]

    # Define styles
    markers = ['o', 's', '^', 'D', 'X']
    dashes = [(1, 0), (4, 1.5), (1, 1), (3, 1, 1, 1), (5, 1, 1, 1, 1, 1)]
    palette = sns.color_palette("bright", n_colors=len(target_deltas))
    
    gammas = sorted(df_filtered['gamma'].unique())
    
    for gamma in gammas:
        subset = df_filtered[df_filtered['gamma'] == gamma]
        if subset.empty:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        
        # Determine common y-limit
        y_max = subset['total_time_s'].max() * 1.1

        for i, version in enumerate(['Old', 'New']):
            ax = axes[i]
            version_subset = subset[subset['version'] == version]
            
            if version_subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{version} Results')
                continue

            sns.lineplot(
                data=version_subset, 
                x='note_count', 
                y='total_time_s', 
                hue='delta', 
                hue_order=delta_order,
                style='delta', 
                style_order=delta_order,
                markers=markers, 
                dashes=dashes, 
                palette=palette,
                linewidth=2,
                markersize=8,
                ax=ax
            )
            
            ax.set_title(f'{version} Results (Fixed $\gamma$={gamma})')
            ax.set_xlabel('Number of Notes')
            if i == 0:
                ax.set_ylabel('Total Runtime (s)')
            else:
                ax.set_ylabel('')
            
            ax.set_ylim(bottom=0, top=y_max)
            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.3)
            ax.legend(title='$\delta$')

        plt.tight_layout()
        output_path = os.path.join(PLOTS_DIR, f'runtime_comparison_fixed_gamma_{gamma}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

def main():
    print("Loading data...")
    df_new = load_data(RESULTS_DIR, 'New')
    df_old = load_data(BACKUP_DIR, 'Old')
    
    if df_new.empty and df_old.empty:
        print("No data found.")
        return

    df = pd.concat([df_old, df_new], ignore_index=True)
    print(f"Loaded {len(df)} rows of data.")
    
    print("Generating comparison plots for fixed Delta...")
    plot_fixed_delta(df)
    
    print("Generating comparison plots for fixed Gamma...")
    plot_fixed_gamma(df)
    
    print("Done.")

if __name__ == "__main__":
    main()
