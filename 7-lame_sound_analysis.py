import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def get_features_from_dict(d, prefix=''):
    """
    Recursively flatten dictionary and compute stats for lists.
    Copied from 6-classification.py to ensure feature consistency.
    """
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']: # Skip metadata
            continue
        
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            # check if list of numbers
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_mean"] = np.mean(v)
                features[f"{prefix}{k}_std"] = np.std(v)
            else:
                pass
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = v
    return features

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '6-classified_gait_features.json')
    output_dir = os.path.join(base_dir, 'lame_sound_analysis_plots')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run 6-classification.py first.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Prepare data for DataFrame
    records = []
    
    # Process Lame
    for vid, features in data.get('lame', {}).items():
        flat_feats = get_features_from_dict(features)
        flat_feats['VideoID'] = vid
        flat_feats['Label'] = 'Lame'
        records.append(flat_feats)
        
    # Process Sound
    for vid, features in data.get('sound', {}).items():
        flat_feats = get_features_from_dict(features)
        flat_feats['VideoID'] = vid
        flat_feats['Label'] = 'Sound'
        records.append(flat_feats)
        
    if not records:
        print("No data found to analyze.")
        return

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} samples. Features found: {len(df.columns) - 2}")
    
    # Identify feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in ['VideoID', 'Label']]
    
    # Statistical Analysis and Plotting
    results = []
    
    print("Performing analysis and generating plots...")
    
    # Set plot style manually since seaborn is missing
    # plt.style.use('ggplot') # Optional, sticking to default with grid
    
    for col in feature_cols:
        # 1. Statistical Test (Mann-Whitney U test)
        lame_vals = df[df['Label'] == 'Lame'][col].dropna()
        sound_vals = df[df['Label'] == 'Sound'][col].dropna()
        
        if len(lame_vals) < 2 or len(sound_vals) < 2:
            continue
            
        stat, p_val = stats.mannwhitneyu(lame_vals, sound_vals, alternative='two-sided')
        
        mean_lame = lame_vals.mean()
        mean_sound = sound_vals.mean()
        
        results.append({
            'Feature': col,
            'P-Value': p_val,
            'Mean_Lame': mean_lame,
            'Mean_Sound': mean_sound,
            'Diff_Mean': mean_lame - mean_sound
        })
        
        # 2. Box Plot with Matplotlib
        # Create a filename safe version of the column name
        safe_col_name = "".join([c if c.isalnum() or c in ('_', '-') else '_' for c in col])
        
        plt.figure(figsize=(8, 6))
        
        # Prepare data list
        data_to_plot = [lame_vals, sound_vals]
        labels = ['Lame', 'Sound']
        
        # Create boxplot
        # patch_artist=True allows filling color
        bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5, showfliers=False)
        
        # Set colors
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        # Jitter plot (Strip plot equivalent)
        for i, vals in enumerate(data_to_plot):
            y = vals
            # Add random jitter to x
            x = np.random.normal(i + 1, 0.04, size=len(y))
            plt.scatter(x, y, alpha=0.6, color='black', s=20, zorder=3)
            
        # Add title with p-value
        significance = ""
        if p_val < 0.001: significance = "***"
        elif p_val < 0.01: significance = "**"
        elif p_val < 0.05: significance = "*"
        
        plt.title(f"{col}\nMann-Whitney p={p_val:.4f} {significance}")
        plt.ylabel("Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{safe_col_name}.png")
        plt.savefig(plot_path)
        plt.close()

    # Save summary statistics
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='P-Value')
        csv_path = os.path.join(base_dir, 'lame_sound_feature_analysis.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nAnalysis complete. Summary saved to: {csv_path}")
        print(f"Plots saved to: {output_dir}")
        
        print("\nTop 10 Most Significant Features:")
        print(results_df.head(10)[['Feature', 'P-Value', 'Mean_Lame', 'Mean_Sound']].to_string(index=False))
    else:
        print("No valid features analyzed.")

if __name__ == "__main__":
    main()
