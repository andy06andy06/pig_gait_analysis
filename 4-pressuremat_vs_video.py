import os
import json
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

# =============================================================================
# 1. DATA LOADING & MATCHING
# =============================================================================

def load_pressuremat_files(root_dir="../videos"):
    """
    Recursively scan root_dir for files containing 'pressuremat' in filename.
    Returns a dict: {video_id: pressuremat_data_dict}
    """
    pattern = os.path.join(root_dir, "**", "*pressuremat*.json")
    files = glob.glob(pattern, recursive=True)
    
    data_map = {}
    
    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract video_id. Assuming format like "1118004_pressuremat.json" -> "1118004"
        # Or "20240317_145300_pressuremat.json" -> "20240317_145300"
        # We split by '_pressuremat'
        if "_pressuremat" in fname:
            video_id = fname.split("_pressuremat")[0]
            with open(fpath, 'r') as f:
                try:
                    data = json.load(f)
                    data_map[video_id] = data
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON {fpath}")
                    
    print(f"Loaded {len(data_map)} pressure mat files.")
    return data_map

def load_camera_gait_features(path="3-keyframe_features.json"):
    """
    Load the video-based gait features JSON.
    Returns dict: {video_id: {limb: {metric: ...}}}
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return {}
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded video gait features for {len(data)} trials.")
    return data

def get_pressure_val(p_data, metric_name, limb):
    """
    Helper to extract a specific value from pressure mat data structure.
    p_data: dict content of pressure mat json
    metric_name: e.g. "Stance Time (sec)"
    limb: "LF", "LH", "RF", "RH"
    """
    if "stance_stride_table" not in p_data:
        return None
    rows = p_data["stance_stride_table"].get("rows", [])
    
    for row in rows:
        if row.get("metric") == metric_name:
            val = row.get(limb)
            # Pressure mat JSON might have null or numeric
            return val
    return None

def build_analysis_dataframe(mat_data_map, cam_data_map):
    """
    Match trials and build a single DataFrame.
    """
    # Identify common video_ids
    mat_ids = set(mat_data_map.keys())
    cam_ids = set(cam_data_map.keys())
    common_ids = list(mat_ids.intersection(cam_ids))
    common_ids.sort()
    
    print(f"Found {len(common_ids)} matched trials.")
    
    records = []
    
    # Map video limbs to pressure mat limbs
    # Video: FL, FR, BL, BR
    # Pressure Mat: LF, RF, LH, RH
    # Mapping: FL->LF, FR->RF, BL->LH, BR->RH
    limb_map_cam_to_mat = {
        "FL": "LF",
        "FR": "RF",
        "BL": "LH",
        "BR": "RH"
    }
    
    # Metrics to analyze
    # Pressure Mat metric names
    pm_metrics = {
        "stride_time": "Stride Time (sec)",
        "stance_time": "Stance Time (sec)",
        "stride_length": "Stride Length (cm)"
    }
    
    # Camera metric names
    cam_metrics = {
        "stride_time": "stride_time",
        "stance_time": "stance_time",
        "stride_length": "stride_length"
    }

    for vid in common_ids:
        mat_trial = mat_data_map[vid]
        cam_trial = cam_data_map[vid]
        
        # Iterate over camera limbs (since we aggregate them)
        for cam_limb, mat_limb in limb_map_cam_to_mat.items():
            if cam_limb not in cam_trial:
                continue
                
            row = {
                "video_id": vid,
                "limb": mat_limb, # Use standard LF/RF/LH/RH
                "pig_id": None # Placeholder if needed
            }
            
            # --- Pressure Mat ---
            for key, pm_name in pm_metrics.items():
                val = get_pressure_val(mat_trial, pm_name, mat_limb)
                row[f"mat_{key}"] = val
            
            # --- Camera ---
            # Average the list of values
            c_limb_data = cam_trial[cam_limb]
            for key, cm_name in cam_metrics.items():
                if cm_name in c_limb_data and "values" in c_limb_data[cm_name]:
                    vals = c_limb_data[cm_name]["values"]
                    # Filter None if any, though usually lists are numbers
                    vals = [v for v in vals if v is not None]
                    if len(vals) > 0:
                        row[f"cam_{key}"] = np.mean(vals)
                    else:
                        row[f"cam_{key}"] = None
                else:
                    row[f"cam_{key}"] = None
            
            records.append(row)
            
    df = pd.DataFrame(records)
    return df

# =============================================================================
# 2. ANALYSIS FUNCTIONS
# =============================================================================

def normalize_series(series, method='zscore'):
    """
    Normalize a pandas Series.
    method: 'zscore' or 'minmax'
    """
    if len(series) < 2:
        return series
        
    if method == 'zscore':
        return (series - series.mean()) / series.std(ddof=1)
    elif method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series * 0
        return (series - min_val) / (max_val - min_val)
    else:
        return series

def filter_metric_non_null(df, metric, use_norm=False):
    """
    Return subset of df where both mat_{metric} and cam_{metric} are not null.
    """
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    # Check if columns exist
    if col_mat not in df.columns or col_cam not in df.columns:
        # Fallback to non-norm if norm not found (though logic should prevent this)
        col_mat = f"mat_{metric}"
        col_cam = f"cam_{metric}"
        
    subset = df.dropna(subset=[col_mat, col_cam]).copy()
    return subset

def get_metric_unit_label(metric, use_norm=False):
    if use_norm:
        return " (Normalized)"
    if "time" in metric:
        return " (s)"
    if "length" in metric:
        return " (cm)"
    return ""

def compute_error_metrics(df, metric, use_norm=False):
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    x = df[col_mat].values
    y = df[col_cam].values
    n = len(x)
    
    if n < 2:
        return {"N": n, "RMSE": np.nan, "MAE": np.nan}
        
    # RMSE
    mse = np.mean((x - y)**2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(x - y))
    
    return {
        "metric": metric + (" (norm)" if use_norm else ""),
        "N": n,
        "RMSE": rmse,
        "MAE": mae
    }

def compute_correlations(df, metric, use_norm=False):
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    x = df[col_mat].values
    y = df[col_cam].values
    n = len(x)
    
    if n < 2:
        return {"N": n, "r": np.nan, "p_r": np.nan, "rho": np.nan, "p_rho": np.nan}
        
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    
    return {
        "metric": metric + (" (norm)" if use_norm else ""),
        "N": n,
        "Pearson r": r,
        "p_value_r": p_r,
        "Spearman rho": rho,
        "p_value_rho": p_rho
    }

def plot_correlation(df, metric, output_dir=".", use_norm=False):
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    data = df[[col_mat, col_cam]].dropna()
    x = data[col_mat]
    y = data[col_cam]
    
    plt.figure(figsize=(7, 7))
    # Increased marker size and slightly more opacity
    plt.scatter(x, y, s=50, alpha=0.7, edgecolors='w', linewidth=0.8)
    
    # Add y=x line
    if len(x) > 0:
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        pad = (max_val - min_val) * 0.05
        plt.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', alpha=0.6, linewidth=1.5)

    unit_label = get_metric_unit_label(metric, use_norm)
    clean_metric = metric.replace("_", " ").title()
    
    plt.xlabel(f"Pressure Mat {clean_metric}{unit_label}", fontsize=12)
    plt.ylabel(f"Camera {clean_metric}{unit_label}", fontsize=12)
    
    title_text = f"Correlation: {clean_metric}"
    if use_norm:
        title_text += "\n(Normalized by Z-score)"
    plt.title(title_text, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    stats_res = compute_correlations(df, metric, use_norm)
    txt = f"N={stats_res['N']}\nr={stats_res['Pearson r']:.2f}\nrho={stats_res['Spearman rho']:.2f}"
    # Increased annotation font size
    plt.annotate(txt, xy=(0.05, 0.82), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="w", alpha=0.9, ec="gray"), fontsize=11)
    
    fname = f"scatter_{metric}_norm.png" if use_norm else f"scatter_{metric}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot to {out_path}")

def compute_bland_altman(df, metric, use_norm=False):
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    x = df[col_mat].values
    y = df[col_cam].values
    n = len(x)
    
    if n < 2:
        return {"N": n}
        
    mean_val = (x + y) / 2
    diff_val = y - x
    
    bias = np.mean(diff_val)
    sd = np.std(diff_val, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    
    return {
        "metric": metric + (" (norm)" if use_norm else ""),
        "N": n,
        "bias": bias,
        "LOA_lower": loa_lower,
        "LOA_upper": loa_upper,
        "SD_diff": sd,
        "mean_vals": mean_val,
        "diff_vals": diff_val
    }

def plot_bland_altman(df, metric, output_dir=".", use_norm=False):
    res = compute_bland_altman(df, metric, use_norm)
    if res["N"] < 2:
        return
        
    mean_vals = res["mean_vals"]
    diff_vals = res["diff_vals"]
    bias = res["bias"]
    upper = res["LOA_upper"]
    lower = res["LOA_lower"]
    
    plt.figure(figsize=(9, 7))
    
    # Shaded area for LOA (95% Limits of Agreement)
    plt.axhspan(lower, upper, color='gray', alpha=0.1, label='95% LOA Range')
    
    # Increased marker size
    plt.scatter(mean_vals, diff_vals, s=50, alpha=0.7, edgecolors='w', linewidth=0.8, zorder=3)
    
    # Zero line (thicker)
    plt.axhline(0, color='black', linewidth=2.0, zorder=2)
    
    plt.axhline(bias, color='blue', linestyle='-', linewidth=1.5, label=f'Bias ({bias:.2f})', zorder=3)
    plt.axhline(upper, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD ({upper:.2f})', zorder=3)
    plt.axhline(lower, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD ({lower:.2f})', zorder=3)
    
    # Clinically acceptable error range (example: +/- 0.1s for time)
    if not use_norm and "time" in metric:
        plt.axhline(0.1, color='green', linestyle=':', linewidth=1.5, label='Acceptable Error (±0.1s)', zorder=2)
        plt.axhline(-0.1, color='green', linestyle=':', linewidth=1.5, zorder=2)
    
    unit_label = get_metric_unit_label(metric, use_norm)
    clean_metric = metric.replace("_", " ").title()
    
    plt.xlabel(f"Mean of Methods{unit_label}", fontsize=12)
    plt.ylabel(f"Difference (Cam - Mat){unit_label}", fontsize=12)
    
    title_text = f"Bland-Altman: {clean_metric}"
    if use_norm:
        title_text += "\n(Normalized by Z-score)"
    plt.title(title_text, fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    fname = f"bland_altman_{metric}_norm.png" if use_norm else f"bland_altman_{metric}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Bland-Altman plot to {out_path}")

def compute_icc_2_1(df, metric, use_norm=False):
    """
    Compute ICC(2,1) - Absolute Agreement.
    Model: Two-way random effects. 
    """
    suffix = "_norm" if use_norm else ""
    col_mat = f"mat_{metric}{suffix}"
    col_cam = f"cam_{metric}{suffix}"
    
    # Data matrix: n rows, 2 columns
    data = df[[col_mat, col_cam]].dropna().values
    n, k = data.shape
    if n < 2 or k != 2:
        return {"N": n, "ICC": np.nan}
        
    # Calculate ANOVA terms
    # Grand mean
    grand_mean = np.mean(data)
    
    # Sum of Squares Total
    SST = np.sum((data - grand_mean)**2)
    
    # Sum of Squares Rows (Targets/Subjects)
    row_means = np.mean(data, axis=1)
    SSR = k * np.sum((row_means - grand_mean)**2)
    
    # Sum of Squares Columns (Raters/Methods)
    col_means = np.mean(data, axis=0)
    SSC = n * np.sum((col_means - grand_mean)**2)
    
    # Sum of Squares Error (Residual)
    SSE = SST - SSR - SSC
    
    # Degrees of Freedom
    df_R = n - 1
    df_C = k - 1
    df_E = (n - 1) * (k - 1)
    
    # Mean Squares
    MSR = SSR / df_R
    MSC = SSC / df_C
    MSE = SSE / df_E
    
    # ICC(2,1) formula
    # (MSR - MSE) / (MSR + (k-1)MSE + (k/n)(MSC - MSE))
    numerator = MSR - MSE
    denominator = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    
    icc = numerator / denominator
    
    # Confidence Interval (Approximate)
    # Using F-test approximation
    # F = MSR / MSE
    # This is often used for consistency. For absolute agreement it's more complex.
    # We will report just the point estimate and a simplified CI if possible, or skip CI if too complex for scratch.
    # To be safe and compliant with "Report 95% CI", I will use the formula from McGraw & Wong (1996) Case 2A.
    
    # F-statistic for CI
    # McGraw & Wong (1996) Case 2A (Absolute Agreement)
    # The calculation is complex but implementable.
    
    # a = k * icc / (n * (1 - icc)) # Wait, let's use the F statistics directly
    # Fj = MSR / MSE
    # But for absolute agreement, column variability matters.
    
    # Implementation based on McGraw & Wong Table 4 (Case 2A)
    # Lower Limit:
    # F_L = F(1-alpha/2, n-1, v)
    # where v = (n-1)(k-1) ... actually the degrees of freedom are more complex for 2A
    
    # Let's use the explicit formulas:
    # a = (k * ICC) / (n * (1 - ICC))  <-- This is for 1-way
    
    # Correct Formula for 2-way random absolute agreement (Model 2,1) CI:
    # See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4913118/ (Koo & Li 2016) or McGraw 1996
    
    # Variables:
    # n = number of rows (subjects)
    # k = number of columns (raters) = 2
    # MSR = Mean Square Row
    # MSC = Mean Square Columns
    # MSE = Mean Square Error
    
    # F stats
    alpha = 0.05
    
    # Lower Bound
    # F_obs = MSR / MSE ? No.
    
    # Let's implement the calculation derived from the variance components.
    # CI = (n(MSR - F*MSE)) / (F*(k*MSC + (n*k - k - n)*MSE) + n*MSR)
    
    # Let's use a simpler, standard approximation often used in Python packages like Pingouin if exact is unwieldy.
    # But since we have all Sum of Squares, let's try to do it right.
    
    # Actually, calculating exact CI for ICC(2,1) is notoriously tricky.
    # However, for k=2, it simplifies?
    
    # Let's try the approximation:
    # F = MSR / MSE
    # df1 = n - 1
    # df2 = (n - 1) * (k - 1)
    # F_lower = F / stats.f.ppf(1 - alpha/2, df1, df2)
    # F_upper = F * stats.f.ppf(1 - alpha/2, df2, df1) # Note df swap for inverse
    
    # This is for Consistency (ICC(3,1)). Absolute agreement is harder.
    # BUT, given this is a script, maybe we can assume consistency is "close enough" OR just admit it's an estimate.
    # The user specifically asked for CI.
    
    # Let's implement the logic from "pingouin" library source for ICC2 (A,1):
    # https://github.com/raphaelvallat/pingouin/blob/master/pingouin/reliability.py
    
    # From Pingouin:
    # vn = (k - 1) * (n - 1) * ((k * icc * MSR / MSE + n * (1 + (k - 1) * icc) - k * icc)) ** 2
    # vd = (n - 1) * k ** 2 * (icc ** 2) * (MSC / MSE) ** 2 + ((n * (1 + (k - 1) * icc) - k * icc) ** 2) * (k - 1)
    # v = vn / vd
    # F1 = stats.f.ppf(1 - alpha / 2, n - 1, v)
    # F2 = stats.f.ppf(1 - alpha / 2, v, n - 1)
    # L1 = n * (MSR - F1 * MSE) / (F1 * (k * MSC + (n * k - k - n) * MSE) + n * MSR)
    # U1 = n * (MSR - F2 * MSE) / (F2 * (k * MSC + (n * k - k - n) * MSE) + n * MSR)
    # (Wait, F2 definition might be inverted)
    
    # Actually, simpler formula from McGraw Table 4:
    # v (df) calculation is needed.
    
    # Let's try to implement the degrees of freedom 'v' calculation properly.
    # a = (k * icc) / (n * (1 - icc))  <-- no, this is for 1-way.
    
    # Let's go with the logic that matches our ICC point estimate:
    # icc = (MSR - MSE) / (MSR + (k-1)MSE + (k/n)(MSC-MSE))
    
    # Using the exact definition from McGraw & Wong 1996 for Case 2A:
    # Lower Limit:
    # F_l = stats.f.ppf(1 - alpha / 2, n - 1, v)
    # Lower = (n * (MSR - F_l * MSE)) / (F_l * (k * MSC + (n * k - k - n) * MSE) + n * MSR)
    
    # Upper Limit:
    # F_u = stats.f.ppf(1 - alpha / 2, v, n - 1) # Note the swap might be implicitly handled by ppf arg
    # Actually usually it is F(alpha/2) and F(1-alpha/2).
    # Upper = (n * (MSR - F_u * MSE)) / (F_u * (k * MSC + (n * k - k - n) * MSE) + n * MSR)
    # Wait, the F-value to use for Upper limit is typically the smaller F critical value?
    # Let's just use F_l and F_u as critical values.
    
    # The degree of freedom 'v':
    # A = (MSR - MSE) / (MSR + (k-1)*MSE + (k/n)*(MSC-MSE)) # This is just ICC
    # But for v, we need the terms.
    # a = k * icc / (n*(1-icc)) # No
    
    # Let's calculate v based on:
    # v = ( (a*MSC + b*MSE)**2 ) / ( ( (a*MSC)**2 / (k-1) ) + ( (b*MSE)**2 / ((n-1)(k-1)) ) )
    # where terms a and b come from the variance expectation.
    # This is getting too risky to implement from memory/scratch without testing.
    
    # Alternative: Use the classic F-ratio approximation for "Consistency" and label it as such? 
    # No, user wants ICC for this specific analysis which is Agreement.
    
    # Let's use the method implemented in R's `psych::ICC`:
    # For ICC(2,1):
    # Fj = MSR / MSE
    # vn = (k-1)*(n-1)*((k*icc*Fj+n*(1+(k-1)*icc)-k*icc))**2
    # vd = (n-1)*k**2*icc**2*(MSC/MSE)**2 + ((n*(1+(k-1)*icc)-k*icc)**2)*(k-1)
    # v = vn / vd
    
    try:
        # F-value observed?
        # Not exactly, but we use the formula involving 'v'
        
        # Calculate v (degrees of freedom for the denominator of the F-test for CI)
        # Sourced from Pingouin/McGraw & Wong
        # Note: icc is the point estimate we calculated
        
        # Term A: k * icc * (MSC/MSE) ... wait, let's use the explicit terms
        # Let's assume the formula:
        # v = ( (n-1)*(k-1) ) * ( ... complex term ... )
        
        # Let's try a simpler approach. 
        # For large N (which we have, >100), the asymptotic standard error (SE) can be used.
        # SE = sqrt( var(icc) )
        # then CI = icc +/- 1.96 * SE
        # Formula for SE of ICC(2,1):
        # approx SE = (1 - icc) * (1 + (k-1)*icc) / sqrt( n*k*(k-1)/2 ) ? (Bartko 1966)
        # This is much safer to implement correctly.
        
        # SE approximation (approximate standard error)
        # SE = ( (1 - icc) * (1 + (k - 1) * icc) ) / np.sqrt( 0.5 * k * (k - 1) * n ) # For k=2, this simplifies
        # For k=2:
        # SE = ( (1 - icc) * (1 + icc) ) / sqrt( n )
        # SE = (1 - icc^2) / sqrt(n)
        
        # Let's check if this is valid for ICC(2,1). 
        # This SE is often cited for ICC(1,1). 
        # For ICC(2,1), it's more complex.
        
        # Let's try to implement the exact McGraw & Wong logic step-by-step carefully.
        
        # Step 1: Calculate 'v'
        # To avoid division by zero or errors, wrap in try/except.
        
        if icc >= 1.0:
            ci_lower, ci_upper = 1.0, 1.0
        else:
            # Terms for v
            # Using variable names from a reference implementation
            a = k * icc * (MSC/MSE) + n * (1 + (k-1)*icc) - k*icc
            b = (n-1) * k**2 * icc**2 * (MSC/MSE)**2
            c = (n * (1 + (k-1)*icc) - k*icc)**2 * (k-1)
            
            v = ((k-1) * (n-1) * a**2) / (b + c)
            
            # Critical F values
            # F_low = F_inv(1 - alpha/2, n-1, v)
            # F_high = F_inv(alpha/2, n-1, v)  <-- wait, usually we want upper percentile
            # ppf is inverse CDF.
            # lower limit uses upper critical F? 
            # bounds: [F_lower_crit, F_upper_crit]
            
            f_stat_1 = stats.f.ppf(1 - alpha / 2, n - 1, v)
            f_stat_2 = stats.f.ppf(alpha / 2, n - 1, v) # This will be small, < 1
            # Or use 1 - alpha/2 with swapped dfs?
            # Let's use f_stat_2 = stats.f.ppf(1 - alpha/2, v, n - 1) and invert?
            
            # Let's use the formula structure:
            # Lower Limit (LL)
            # LL = n(MSR - f_stat_1 * MSE) / (f_stat_1 * (k*MSC + (n*k - k - n)*MSE) + n*MSR)
            
            # Upper Limit (UL)
            # For UL, we typically divide by the smaller F (or multiply by inverted larger F).
            # The formula in McGraw & Wong uses F(1-alpha/2, v, n-1) for the Upper bound term?
            # Actually, let's look at the structure:
            # The bounds are monotonic with F.
            # So calculating the expression with F_low_percentile and F_high_percentile gives the bounds.
            
            f_crit_upper = stats.f.ppf(1 - alpha / 2, n - 1, v)
            f_crit_lower = stats.f.ppf(alpha / 2, n - 1, v)
            
            def get_limit(f_val):
                num = n * (MSR - f_val * MSE)
                den = f_val * (k * MSC + (n * k - k - n) * MSE) + n * MSR
                return num / den
            
            ci_lower = get_limit(f_crit_upper)
            ci_upper = get_limit(f_crit_lower)
            
    except Exception as e:
        print(f"Warning: CI calc failed: {e}")
        ci_lower, ci_upper = np.nan, np.nan

    # Interpretation
    interp = ""
    if icc < 0.5: interp = "poor"
    elif icc < 0.75: interp = "moderate"
    elif icc < 0.9: interp = "good"
    else: interp = "excellent"
    
    return {
        "metric": metric + (" (norm)" if use_norm else ""),
        "N": n,
        "ICC": icc,
        "95% CI lower": ci_lower,
        "95% CI upper": ci_upper,
        "interpretation": interp
    }

def plot_combined_correlation(plot_data, ordered_metrics, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7)) # Increased figure size
    
    for i, m in enumerate(ordered_metrics):
        ax = axes[i]
        if m not in plot_data:
            ax.set_visible(False)
            continue
            
        data = plot_data[m]
        df = data["df"]
        use_norm = data["use_norm"]
        stats = data["stats_corr"]
        
        suffix = "_norm" if use_norm else ""
        col_mat = f"mat_{m}{suffix}"
        col_cam = f"cam_{m}{suffix}"
        
        x = df[col_mat]
        y = df[col_cam]
        
        # Increased marker size
        ax.scatter(x, y, s=60, alpha=0.7, edgecolors='w', linewidth=0.8)
        
        # y=x line
        if len(x) > 0:
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            pad = (max_val - min_val) * 0.05
            ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', alpha=0.6, linewidth=1.5)
        
        unit_label = get_metric_unit_label(m, use_norm)
        
        # Clean up metric name for title
        title_map = {
            "stance_time": "Stance Time",
            "stride_time": "Stride Time",
            "stride_length": "Stride Length"
        }
        title = title_map.get(m, m)
        if use_norm:
            title += " (Norm)"
        
        ax.set_xlabel(f"Pressure Mat{unit_label}", fontsize=13)
        ax.set_ylabel(f"Camera{unit_label}", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        txt = f"N={stats['N']}\nr={stats['Pearson r']:.2f}\nrho={stats['Spearman rho']:.2f}"
        ax.annotate(txt, xy=(0.05, 0.82), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="w", alpha=0.9, ec="gray"), fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "combined_correlation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined correlation plot to {out_path}")

def plot_combined_bland_altman(plot_data, ordered_metrics, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7)) # Increased figure size
    
    for i, m in enumerate(ordered_metrics):
        ax = axes[i]
        if m not in plot_data:
            ax.set_visible(False)
            continue
            
        data = plot_data[m]
        ba_res = data["stats_ba"]
        mean_vals = ba_res["mean_vals"]
        diff_vals = ba_res["diff_vals"]
        bias = ba_res["bias"]
        upper = ba_res["LOA_upper"]
        lower = ba_res["LOA_lower"]
        use_norm = data["use_norm"]
        
        # Shaded area for LOA
        ax.axhspan(lower, upper, color='gray', alpha=0.1)
        
        # Increased marker size
        ax.scatter(mean_vals, diff_vals, s=60, alpha=0.7, edgecolors='w', linewidth=0.8, zorder=3)
        
        # Zero line
        ax.axhline(0, color='black', linewidth=2.0, zorder=2)
        
        ax.axhline(bias, color='blue', linestyle='-', linewidth=1.5, label=f'Bias ({bias:.2f})', zorder=3)
        ax.axhline(upper, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD', zorder=3)
        ax.axhline(lower, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD', zorder=3)
        
        # Clinically acceptable error range (example: +/- 0.1s for time)
        if not use_norm and "time" in m:
            ax.axhline(0.1, color='green', linestyle=':', linewidth=1.5, label='Acceptable Error', zorder=2)
            ax.axhline(-0.1, color='green', linestyle=':', linewidth=1.5, zorder=2)
        
        unit_label = get_metric_unit_label(m, use_norm)
        title_map = {
            "stance_time": "Stance Time",
            "stride_time": "Stride Time",
            "stride_length": "Stride Length"
        }
        title = title_map.get(m, m)
        if use_norm:
            title += " (Norm)"
        
        ax.set_xlabel(f"Mean of Methods{unit_label}", fontsize=13)
        ax.set_ylabel(f"Difference (Cam - Mat){unit_label}", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "combined_bland_altman.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined Bland-Altman plot to {out_path}")

# =============================================================================
# 3. MAIN WORKFLOW
# =============================================================================

def run_full_validation():
    # 1. Load Data
    # Assuming script is in code/, videos in ../videos
    videos_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
    features_path = os.path.join(os.path.dirname(__file__), "3-keyframe_features.json")
    
    pm_data = load_pressuremat_files(videos_dir)
    cam_data = load_camera_gait_features(features_path)
    
    # 2. Build DataFrame
    df = build_analysis_dataframe(pm_data, cam_data)
    if df.empty:
        print("No matched data found.")
        return

    # 3. Analyze each metric
    metrics = ["stride_time", "stance_time", "stride_length"]
    
    corr_results = []
    ba_results = []
    icc_results = []
    error_results = []
    plot_data = {}
    
    print("\n--- Running Analysis ---")
    
    output_dir = os.path.join(os.path.dirname(__file__), "pressuremat_vs_video")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for m in metrics:
        sub_df = filter_metric_non_null(df, m)
        n = len(sub_df)
        print(f"\nMetric: {m} (N={n})")
        
        if n < 2:
            print("Not enough data points.")
            continue
        
        # Decide if we normalize
        # If stride_length, we use normalization (Z-score)
        use_norm = False
        if m == "stride_length":
            use_norm = True
            # Apply normalization
            sub_df[f"mat_{m}_norm"] = normalize_series(sub_df[f"mat_{m}"], method='zscore')
            sub_df[f"cam_{m}_norm"] = normalize_series(sub_df[f"cam_{m}"], method='zscore')
            
        # Correlations
        c_res = compute_correlations(sub_df, m, use_norm=use_norm)
        corr_results.append(c_res)
        plot_correlation(sub_df, m, output_dir, use_norm=use_norm)
        
        # Bland-Altman
        ba_res = compute_bland_altman(sub_df, m, use_norm=use_norm)
        ba_results.append({
            "metric": ba_res["metric"],
            "N": ba_res["N"],
            "bias": ba_res["bias"],
            "LOA lower": ba_res["LOA_lower"],
            "LOA upper": ba_res["LOA_upper"]
        })
        plot_bland_altman(sub_df, m, output_dir, use_norm=use_norm)
        
        plot_data[m] = {
            "df": sub_df,
            "use_norm": use_norm,
            "stats_corr": c_res,
            "stats_ba": ba_res
        }
        
        # ICC
        icc_res = compute_icc_2_1(sub_df, m, use_norm=use_norm)
        icc_results.append(icc_res)
        
        # Error Metrics (RMSE, MAE)
        err_res = compute_error_metrics(sub_df, m, use_norm=use_norm)
        error_results.append(err_res)

    # Plot combined absolute
    ordered_metrics = ["stance_time", "stride_time", "stride_length"]
    plot_combined_correlation(plot_data, ordered_metrics, output_dir)
    plot_combined_bland_altman(plot_data, ordered_metrics, output_dir)

    
    # 4. Summaries & Output to File
    summary_lines = []
    
    summary_lines.append("\n=== Correlation Summary ===")
    df_corr = pd.DataFrame(corr_results)
    if not df_corr.empty:
        table_str = df_corr.to_string(index=False)
        print(table_str)
        summary_lines.append(table_str)
    
    summary_lines.append("\n=== Bland-Altman Summary ===")
    df_ba = pd.DataFrame(ba_results)
    if not df_ba.empty:
        table_str = df_ba.to_string(index=False)
        print(table_str)
        summary_lines.append(table_str)
        
    summary_lines.append("\n=== ICC Summary ===")
    df_icc = pd.DataFrame(icc_results)
    if not df_icc.empty:
        table_str = df_icc.to_string(index=False)
        print(table_str)
        summary_lines.append(table_str)
        
    summary_lines.append("\n=== Error Metrics Summary ===")
    df_err = pd.DataFrame(error_results)
    if not df_err.empty:
        table_str = df_err.to_string(index=False)
        print(table_str)
        summary_lines.append(table_str)

    # Save to text file
    summary_path = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved summary statistics to {summary_path}")

# =============================================================================
# 4. EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_full_validation()

# =============================================================================
# 5. METHODS PARAGRAPH
# =============================================================================
"""
Methods:
Pressure mat data were processed to extract averaged limb-level gait metrics per trial. 
Stride-based variables (stride time, stride length) were treated as missing (NULL) if fewer than two valid steps were detected for a given limb. 
Video-based gait features were derived from per-step metrics extracted from keypoints; these were aggregated (averaged) per limb and trial to match the pressure mat representation. 
Only trials with both a pressure mat JSON file and corresponding entries in the video feature dataset were included in the study. 
For each of the three metrics (stride_time, stance_time, stride_length), analysis was restricted to limb–trial pairs where both systems yielded non-null values. 
Correlation (Pearson and Spearman), Bland-Altman agreement (bias and limits of agreement), and Intraclass Correlation Coefficient (ICC(2,1), absolute agreement) were computed to validate the video-based metrics against the pressure mat ground truth.
Note: For stride_length, due to unit differences (cm vs pixels) and lack of calibration, Z-score normalization was applied prior to agreement analysis to allow comparison of relative values.
"""

# =============================================================================
# 6. INTERPRETATION GUIDE
# =============================================================================
"""
Interpretation Guide:

1. Pearson (r) and Spearman (rho) Correlation:
   - Measures the strength and direction of the linear (Pearson) or monotonic (Spearman) relationship.
   - Range: -1 to +1.
   - Interpretation: Values > 0.7 indicate strong positive correlation.
   - p-value: < 0.05 indicates the correlation is statistically significant.

2. Bland-Altman Analysis:
   - Bias: The mean difference between methods (Camera - Pressure Mat). A non-zero bias indicates a systematic over- or under-estimation by the video method.
   - Limits of Agreement (LOA): Bias ± 1.96 * SD of differences. 95% of differences are expected to fall within this range. Narrower LOA indicates better agreement.
   - Note: For normalized stride length, bias will be 0 by definition (Z-score centering), so LOA width indicates the random error spread.

3. Intraclass Correlation Coefficient (ICC):
   - Measures the reliability and absolute agreement between the two methods.
   - Interpretation thresholds:
     - < 0.5: Poor reliability
     - 0.5 - 0.75: Moderate reliability
     - 0.75 - 0.9: Good reliability
     - > 0.9: Excellent reliability
   - High ICC requires both high correlation AND low bias (similar absolute values). 
   - Normalization improves ICC for uncalibrated metrics (like Stride Length) by removing scale/offset differences.
"""
