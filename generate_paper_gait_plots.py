#!/usr/bin/env python3
"""
Publication-Quality Gait Feature Analysis & Visualization for Thesis/Paper.

Generates:
1. Figure 1: Spatio-Temporal Gait Patterns Across Limbs & Lameness Levels (2x2 Panel)
2. Figure 2: Postural & Kinematic Biomechanics Across Lameness Levels (2x2 Panel)
3. Figure 3: Limb Symmetry & Bilateral Coordination Breakdown (2x2 Panel)
4. Figure 4: Global Feature Correlation Heatmap & 2D PCA Projection (1x2 Wide Panel)
5. Table 1: Comprehensive Statistical Summary Table (CSV & Markdown)

Supports both 3-Level (Sound, Medium, Lame) and 2-Level (Sound, Lame) datasets.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==============================================================================
# Style & Color Palette Configuration (Nature / Science Academic Standard)
# ==============================================================================
PALETTE_3LEVELS = {
    'level1_sound': {'color': '#1f77b4', 'label': 'Level 1: Sound', 'name': 'Sound', 'light': '#aec7e8'},
    'level2_medium': {'color': '#ff7f0e', 'label': 'Level 2: Medium', 'name': 'Medium', 'light': '#ffbb78'},
    'level3_lame': {'color': '#d62728', 'label': 'Level 3: Lame', 'name': 'Lame', 'light': '#ff9896'}
}

PALETTE_2LEVELS = {
    'sound': {'color': '#1f77b4', 'label': 'Sound', 'name': 'Sound', 'light': '#aec7e8'},
    'lame': {'color': '#d62728', 'label': 'Lame', 'name': 'Lame', 'light': '#ff9896'}
}

PALETTE_91VIDEOS = {
    'level0_sound': {'color': '#1f77b4', 'label': 'Level 0: Sound', 'name': 'Sound', 'light': '#aec7e8'},
    'level1_medium': {'color': '#ff7f0e', 'label': 'Level 1: Medium', 'name': 'Medium', 'light': '#ffbb78'},
    'level2_severe': {'color': '#d62728', 'label': 'Level 2: Severe', 'name': 'Severe', 'light': '#ff9896'}
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.1,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# ==============================================================================
# Helper Functions: Data Extraction & Preprocessing
# ==============================================================================
def extract_flat_features(d, prefix=''):
    """Recursively extract numeric features from nested JSON dictionary."""
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']:
            continue
        if isinstance(v, dict):
            features.update(extract_flat_features(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_mean"] = float(np.mean(v))
                features[f"{prefix}{k}_std"] = float(np.std(v))
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = float(v)
    return features


def load_dataset(json_path, dataset_type='3level'):
    """Load JSON file and convert into structured Pandas DataFrame."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    records = []
    if dataset_type == '91videos_3level':
        level_order_map = {'level0_sound': 0, 'level1_medium': 1, 'level2_severe': 2}
    elif dataset_type == '3level':
        level_order_map = {'level1_sound': 1, 'level2_medium': 2, 'level3_lame': 3}
    else:
        level_order_map = {'sound': 1, 'lame': 2}

    for group_name, sample_dict in raw_data.items():
        for vid, feat_dict in sample_dict.items():
            row = extract_flat_features(feat_dict)
            row['video_id'] = vid
            row['group'] = group_name
            row['group_rank'] = level_order_map.get(group_name, 0)
            records.append(row)

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# Statistical Annotations & Significance Utilities
# ==============================================================================
def compute_kruskal_p(df, col_name, group_col='group'):
    """Compute Kruskal-Wallis p-value across groups."""
    groups = [vals.dropna().values for _, vals in df.groupby(group_col)[col_name]]
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return np.nan
    all_vals = np.concatenate(valid_groups)
    if len(np.unique(all_vals)) <= 1:
        return np.nan
    try:
        stat, p = stats.kruskal(*valid_groups)
        return p
    except Exception:
        return np.nan


def get_significance_stars(p):
    """Return academic asterisk notation for p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def plot_grouped_boxes_with_points(ax, data_groups, positions, colors, widths=0.22, jitter_strength=0.03):
    """
    Draw publication-quality boxplots overlaid with jittered individual data points.
    data_groups: list of pandas Series/numpy arrays
    positions: list of numeric x coordinates
    colors: list of hex colors
    """
    for data, pos, col in zip(data_groups, positions, colors):
        clean_data = pd.Series(data).dropna().values
        if len(clean_data) <= 2:
            med = np.median(clean_data)
            ax.plot([pos - widths * 0.4, pos + widths * 0.4], [med, med], color='black', linewidth=2.2, zorder=3)
            ax.scatter(
                [pos] * len(clean_data),
                clean_data,
                marker='D',
                color=col,
                edgecolors='black',
                linewidths=1.2,
                s=60,
                alpha=0.95,
                zorder=5
            )
        else:
            bp = ax.boxplot(
                [clean_data],
                positions=[pos],
                widths=widths,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor=col, alpha=0.65, edgecolor='black', linewidth=1.1),
                whiskerprops=dict(color='black', linewidth=1.1, linestyle='--'),
                capprops=dict(color='black', linewidth=1.1)
            )
            rng = np.random.default_rng(int(abs(pos) * 100) + 42)
            jitter = rng.normal(0, jitter_strength, size=len(clean_data))
            ax.scatter(
                pos + jitter,
                clean_data,
                color=col,
                edgecolors='black',
                linewidths=0.6,
                s=28,
                alpha=0.85,
                zorder=4
            )


def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Draw a covariance confidence ellipse on matplotlib axes."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=theta,
                  facecolor=facecolor, **kwargs)
    return ax.add_patch(ell)


# ==============================================================================
# Figure 1: Spatio-Temporal Gait Patterns Across Limbs & Lameness Levels
# ==============================================================================
def generate_figure1_spatiotemporal(df, output_prefix, palette_cfg):
    """
    Panel A: Stance Time across 4 Limbs (FL, FR, BL, BR) by Level
    Panel B: Stride Length across 4 Limbs by Level
    Panel C: Duty Factor (%) across 4 Limbs by Level
    Panel D: Stride Time (Cycle Duration) across 4 Limbs by Level
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig.subplots_adjust(hspace=0.28, wspace=0.24)

    limbs = ['FL', 'FR', 'BL', 'BR']
    limb_names = ['Fore Left (FL)', 'Fore Right (FR)', 'Hind Left (BL)', 'Hind Right (BR)']
    groups = list(palette_cfg.keys())
    n_groups = len(groups)
    
    box_width = 0.22 if n_groups == 3 else 0.30
    offsets = (
        [-box_width * 1.1, 0, box_width * 1.1] if n_groups == 3
        else [-box_width * 0.6, box_width * 0.6]
    )

    panels_config = [
        {
            'ax': axes[0, 0],
            'title': '(A) Stance Phase Duration across Limbs',
            'ylabel': 'Stance Time (seconds)',
            'feat_pattern': '{limb}_stance_time_values_mean',
            'unit_scale': 1.0
        },
        {
            'ax': axes[0, 1],
            'title': '(B) Stride Length across Limbs',
            'ylabel': 'Stride Length (pixels)',
            'feat_pattern': '{limb}_stride_length_values_mean',
            'unit_scale': 1.0
        },
        {
            'ax': axes[1, 0],
            'title': '(C) Duty Factor (Support Phase Proportion)',
            'ylabel': 'Duty Factor (%)',
            'feat_pattern': '{limb}_duty_factor_values_mean',
            'unit_scale': 100.0
        },
        {
            'ax': axes[1, 1],
            'title': '(D) Stride Cycle Duration across Limbs',
            'ylabel': 'Stride Time (seconds)',
            'feat_pattern': '{limb}_stride_time_values_mean',
            'unit_scale': 1.0
        }
    ]

    for pcfg in panels_config:
        ax = pcfg['ax']
        ax.set_title(pcfg['title'], fontweight='bold', pad=10, loc='left')
        ax.set_ylabel(pcfg['ylabel'], fontweight='semibold')
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

        for limb_idx, limb in enumerate(limbs):
            feat_name = pcfg['feat_pattern'].format(limb=limb)
            if feat_name not in df.columns:
                continue

            limb_center = limb_idx * 1.0
            data_list = []
            pos_list = []
            color_list = []

            for g_idx, grp in enumerate(groups):
                subset = df[df['group'] == grp][feat_name].dropna() * pcfg['unit_scale']
                data_list.append(subset)
                pos_list.append(limb_center + offsets[g_idx])
                color_list.append(palette_cfg[grp]['color'])

            plot_grouped_boxes_with_points(ax, data_list, pos_list, color_list, widths=box_width * 0.85)

            p_val = compute_kruskal_p(df, feat_name)
            stars = get_significance_stars(p_val)
            if stars and stars != "ns":
                all_grp_data = pd.concat([d for d in data_list if len(d) > 0])
                if len(all_grp_data) > 0:
                    y_max = all_grp_data.max()
                    y_range = y_max - all_grp_data.min() if y_max > all_grp_data.min() else 1.0
                    ax.text(limb_center, y_max + y_range * 0.08, stars, ha='center', va='bottom',
                            fontweight='bold', color='crimson', fontsize=11)

        if '91videos' in output_prefix:
            if 'Stance Time' in pcfg['ylabel']:
                ax.set_ylim(0, 2.5)
                for limb_idx in range(len(limbs)):
                    ax.text(limb_idx + offsets[-1], 2.40, '7.3s ↑', ha='center', va='top', fontsize=8, color='#d62728', fontweight='bold')
            elif 'Stride Length' in pcfg['ylabel']:
                ax.set_ylim(100, 520)
                for limb_idx in range(len(limbs)):
                    ax.text(limb_idx + offsets[-1], 500, 'Freeze*', ha='center', va='top', fontsize=8, color='#d62728', fontweight='bold')

        ax.set_xticks(range(len(limbs)))
        ax.set_xticklabels(limb_names, fontweight='semibold')

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette_cfg[grp]['color'], edgecolor='black', alpha=0.7)
        for grp in groups
    ]
    labels = [
        f"{palette_cfg[grp]['label']} (n={len(df[df['group'] == grp])})"
        for grp in groups
    ]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=n_groups, frameon=True, facecolor='#fafafa', edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = f"{output_prefix}_Figure1_SpatioTemporal_Dynamics.png"
    pdf_path = f"{output_prefix}_Figure1_SpatioTemporal_Dynamics.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# Figure 2: Postural & Kinematic Biomechanics Across Lameness Levels
# ==============================================================================
def generate_figure2_posture_kinematics(df, output_prefix, palette_cfg):
    """
    Panel A: Back Height Arching Metric (H)
    Panel B: Hoof Release Angles (alpha1 & alpha2)
    Panel C: Neck-Back Posture Angles (beta1 & beta2)
    Panel D: Head Bobbing Vertical Amplitude & Dispersion
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig.subplots_adjust(hspace=0.28, wspace=0.24)

    groups = list(palette_cfg.keys())
    n_groups = len(groups)
    box_width = 0.22 if n_groups == 3 else 0.30
    offsets = (
        [-box_width * 1.1, 0, box_width * 1.1] if n_groups == 3
        else [-box_width * 0.6, box_width * 0.6]
    )

    # Panel A: Back Height Arching Metric (H)
    ax_a = axes[0, 0]
    ax_a.set_title('(A) Back Height Elevation Metric (H)', fontweight='bold', pad=10, loc='left')
    ax_a.set_ylabel('Back Elevation Metric H (pixels)', fontweight='semibold')
    ax_a.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    h_col = 'back_height_feature_H_values_mean'
    if h_col in df.columns:
        data_list = [df[df['group'] == grp][h_col].dropna() for grp in groups]
        pos_list = list(range(n_groups))
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_a, data_list, pos_list, colors, widths=0.45)
        ax_a.set_xticks(range(n_groups))
        ax_a.set_xticklabels([palette_cfg[grp]['name'] for grp in groups], fontweight='semibold')
        p_val = compute_kruskal_p(df, h_col)
        stars = get_significance_stars(p_val)
        p_text = f"p < 0.001 {stars}" if p_val < 0.001 else f"p = {p_val:.3f} {stars}"
        ax_a.text(0.95, 0.92, f"Kruskal-Wallis\n{p_text}", transform=ax_a.transAxes,
                  ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    # Panel B: Hoof Release Angles
    ax_b = axes[0, 1]
    ax_b.set_title(r'(B) Hoof Release Angles ($\alpha_1$ & $\alpha_2$)', fontweight='bold', pad=10, loc='left')
    ax_b.set_ylabel('Angle (degrees)', fontweight='semibold')
    ax_b.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    angle_cols = [
        ('front_hoof_release_angle_alpha1_values_mean', r'Front Hoof ($\alpha_1$)'),
        ('hind_hoof_release_angle_alpha1_values_mean', r'Hind Hoof ($\alpha_1$)'),
        ('front_hoof_release_angle_alpha2_values_mean', r'Front Hoof ($\alpha_2$)'),
    ]

    valid_angles = [item for item in angle_cols if item[0] in df.columns]
    for a_idx, (col_name, col_label) in enumerate(valid_angles):
        center = a_idx * 1.0
        data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_b, data_list, pos_list, colors, widths=box_width * 0.85)

        p_val = compute_kruskal_p(df, col_name)
        stars = get_significance_stars(p_val)
        if stars and stars != "ns":
            all_d = pd.concat([d for d in data_list if len(d) > 0])
            if len(all_d) > 0:
                ax_b.text(center, all_d.max() + (all_d.max() - all_d.min()) * 0.08, stars,
                          ha='center', va='bottom', fontweight='bold', color='crimson', fontsize=11)

    ax_b.set_xticks(range(len(valid_angles)))
    ax_b.set_xticklabels([item[1] for item in valid_angles], fontweight='semibold')

    # Panel C: Neck-Back Posture Angles
    ax_c = axes[1, 0]
    ax_c.set_title(r'(C) Cervicothoracic / Body Posture Angles ($\beta_1$ & $\beta_2$)', fontweight='bold', pad=10, loc='left')
    ax_c.set_ylabel('Angle (degrees)', fontweight='semibold')
    ax_c.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    beta_cols = [
        ('back_neck_angle_beta1_values_mean', r'Neck-Back ($\beta_1$)'),
        ('back_neck_angle_beta2_values_mean', r'Torso Inclination ($\beta_2$)')
    ]

    valid_betas = [item for item in beta_cols if item[0] in df.columns]
    for b_idx, (col_name, col_label) in enumerate(valid_betas):
        center = b_idx * 1.0
        data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_c, data_list, pos_list, colors, widths=box_width * 0.85)

        p_val = compute_kruskal_p(df, col_name)
        stars = get_significance_stars(p_val)
        if stars and stars != "ns":
            all_d = pd.concat([d for d in data_list if len(d) > 0])
            if len(all_d) > 0:
                ax_c.text(center, all_d.max() + (all_d.max() - all_d.min()) * 0.08, stars,
                          ha='center', va='bottom', fontweight='bold', color='crimson', fontsize=11)

    ax_c.set_xticks(range(len(valid_betas)))
    ax_c.set_xticklabels([item[1] for item in valid_betas], fontweight='semibold')

    # Panel D: Head Bobbing or Range of Motion
    ax_d = axes[1, 1]
    head_cols = [
        ('head_bobbing_head_y_amp_values_mean', 'Peak-to-Peak Amplitude'),
        ('head_bobbing_head_y_sd_values_mean', 'Standard Deviation (SD)')
    ]
    available_head = [item for item in head_cols if item[0] in df.columns]
    if available_head:
        ax_d.set_title('(D) Head Bobbing Motion Dynamics (Nodding Compensation)', fontweight='bold', pad=10, loc='left')
        ax_d.set_ylabel('Vertical Displacement (pixels)', fontweight='semibold')
        ax_d.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        for h_idx, (col_name, col_label) in enumerate(available_head):
            center = h_idx * 1.0
            data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
            pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
            colors = [palette_cfg[grp]['color'] for grp in groups]
            plot_grouped_boxes_with_points(ax_d, data_list, pos_list, colors, widths=box_width * 0.85)

            p_val = compute_kruskal_p(df, col_name)
            stars = get_significance_stars(p_val)
            if stars and stars != "ns":
                all_d = pd.concat([d for d in data_list if len(d) > 0])
                if len(all_d) > 0:
                    ax_d.text(center, all_d.max() + (all_d.max() - all_d.min()) * 0.08, stars,
                              ha='center', va='bottom', fontweight='bold', color='crimson', fontsize=11)

        ax_d.set_xticks(range(len(available_head)))
        ax_d.set_xticklabels([item[1] for item in available_head], fontweight='semibold')
    else:
        rom_cols = [
            ('front_hoof_release_angle_alpha1_rom_values_mean', r'Front Hoof $\alpha_1$ ROM'),
            ('front_hoof_release_angle_alpha2_rom_values_mean', r'Front Hoof $\alpha_2$ ROM')
        ]
        available_rom = [item for item in rom_cols if item[0] in df.columns]
        ax_d.set_title('(D) Angular Range of Motion (ROM)', fontweight='bold', pad=10, loc='left')
        ax_d.set_ylabel('ROM Angle (degrees)', fontweight='semibold')
        ax_d.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        for r_idx, (col_name, col_label) in enumerate(available_rom):
            center = r_idx * 1.0
            data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
            pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
            colors = [palette_cfg[grp]['color'] for grp in groups]
            plot_grouped_boxes_with_points(ax_d, data_list, pos_list, colors, widths=box_width * 0.85)
        ax_d.set_xticks(range(len(available_rom)))
        ax_d.set_xticklabels([item[1] for item in available_rom], fontweight='semibold')

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette_cfg[grp]['color'], edgecolor='black', alpha=0.7)
        for grp in groups
    ]
    labels = [
        f"{palette_cfg[grp]['label']} (n={len(df[df['group'] == grp])})"
        for grp in groups
    ]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=n_groups, frameon=True, facecolor='#fafafa', edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = f"{output_prefix}_Figure2_Posture_Kinematics.png"
    pdf_path = f"{output_prefix}_Figure2_Posture_Kinematics.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# Figure 3: Limb Symmetry & Coordination Breakdown
# ==============================================================================
def generate_figure3_symmetry(df, output_prefix, palette_cfg):
    """
    Panel A: Forelimb Left/Right Symmetry Ratios (Stance Time, Stride Length, Stride Time)
    Panel B: Hindlimb Left/Right Symmetry Ratios (Stance Time, Stride Length, Stride Time)
    Panel C: Fore-Hind Load Sharing Ratio (Front / Hind)
    Panel D: Asymmetry Divergence Magnitude (|1 - Ratio|)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig.subplots_adjust(hspace=0.28, wspace=0.24)

    groups = list(palette_cfg.keys())
    n_groups = len(groups)
    box_width = 0.22 if n_groups == 3 else 0.30
    offsets = (
        [-box_width * 1.1, 0, box_width * 1.1] if n_groups == 3
        else [-box_width * 0.6, box_width * 0.6]
    )

    # Panel A: Forelimb Left/Right Symmetry
    ax_a = axes[0, 0]
    ax_a.set_title('(A) Forelimb Left/Right Symmetry Ratios (LF / RF)', fontweight='bold', pad=10, loc='left')
    ax_a.set_ylabel('Symmetry Ratio (LF / RF)', fontweight='semibold')
    ax_a.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax_a.axhline(1.0, color='crimson', linestyle='--', lw=1.3, alpha=0.85, label='Perfect Symmetry (1.0)')
    ax_a.axhspan(0.95, 1.05, facecolor='lightgreen', alpha=0.2, label='Normal Range (±5%)')

    lf_rf_metrics = [
        ('symmetry_ratio_Left Front / Right Front_stance_time', 'Stance Time'),
        ('symmetry_ratio_Left Front / Right Front_stride_length', 'Stride Length'),
        ('symmetry_ratio_Left Front / Right Front_stride_time', 'Stride Time')
    ]

    for m_idx, (col_name, col_label) in enumerate(lf_rf_metrics):
        center = m_idx * 1.0
        if col_name not in df.columns:
            continue
        data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_a, data_list, pos_list, colors, widths=box_width * 0.85)

    ax_a.set_xticks(range(len(lf_rf_metrics)))
    ax_a.set_xticklabels([item[1] for item in lf_rf_metrics], fontweight='semibold')
    ax_a.legend(loc='lower left', fontsize=8.5, framealpha=0.8)

    # Panel B: Hindlimb Left/Right Symmetry
    ax_b = axes[0, 1]
    ax_b.set_title('(B) Hindlimb Left/Right Symmetry Ratios (LH / RH)', fontweight='bold', pad=10, loc='left')
    ax_b.set_ylabel('Symmetry Ratio (LH / RH)', fontweight='semibold')
    ax_b.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax_b.axhline(1.0, color='crimson', linestyle='--', lw=1.3, alpha=0.85)
    ax_b.axhspan(0.95, 1.05, facecolor='lightgreen', alpha=0.2)

    lh_rh_metrics = [
        ('symmetry_ratio_Left Hind / Right Hind_stance_time', 'Stance Time'),
        ('symmetry_ratio_Left Hind / Right Hind_stride_length', 'Stride Length'),
        ('symmetry_ratio_Left Hind / Right Hind_stride_time', 'Stride Time')
    ]

    for m_idx, (col_name, col_label) in enumerate(lh_rh_metrics):
        center = m_idx * 1.0
        if col_name not in df.columns:
            continue
        data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_b, data_list, pos_list, colors, widths=box_width * 0.85)

    ax_b.set_xticks(range(len(lh_rh_metrics)))
    ax_b.set_xticklabels([item[1] for item in lh_rh_metrics], fontweight='semibold')

    # Panel C: Fore-Hind Coordination Ratio
    ax_c = axes[1, 0]
    ax_c.set_title('(C) Fore-Hind Coordination Ratios (Front / Hind)', fontweight='bold', pad=10, loc='left')
    ax_c.set_ylabel('Ratio (Front / Hind)', fontweight='semibold')
    ax_c.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax_c.axhline(1.0, color='crimson', linestyle='--', lw=1.3, alpha=0.85)

    fh_metrics = [
        ('symmetry_ratio_Front / Hind_stance_time', 'Stance Time'),
        ('symmetry_ratio_Front / Hind_stride_length', 'Stride Length'),
        ('symmetry_ratio_Front / Hind_stride_time', 'Stride Time')
    ]

    for m_idx, (col_name, col_label) in enumerate(fh_metrics):
        center = m_idx * 1.0
        if col_name not in df.columns:
            continue
        data_list = [df[df['group'] == grp][col_name].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_c, data_list, pos_list, colors, widths=box_width * 0.85)

    ax_c.set_xticks(range(len(fh_metrics)))
    ax_c.set_xticklabels([item[1] for item in fh_metrics], fontweight='semibold')

    # Panel D: Absolute Direction-Independent Asymmetry Deviation
    ax_d = axes[1, 1]
    ax_d.set_title('(D) Magnitude of Bilateral Asymmetry Deviation (|1 - Ratio|)', fontweight='bold', pad=10, loc='left')
    ax_d.set_ylabel('Absolute Deviation |1 - Ratio|', fontweight='semibold')
    ax_d.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    dev_cols = [
        ('dev_lf_rf_stance', 'symmetry_ratio_Left Front / Right Front_stance_time', 'Forelimb L/R (Stance)'),
        ('dev_lh_rh_stance', 'symmetry_ratio_Left Hind / Right Hind_stance_time', 'Hindlimb L/R (Stance)'),
        ('dev_fh_stance', 'symmetry_ratio_Front / Hind_stance_time', 'Fore / Hind (Stance)')
    ]

    temp_df = df.copy()
    for new_col, src_col, _ in dev_cols:
        if src_col in temp_df.columns:
            temp_df[new_col] = (temp_df[src_col] - 1.0).abs()

    for d_idx, (new_col, src_col, label) in enumerate(dev_cols):
        center = d_idx * 1.0
        if new_col not in temp_df.columns:
            continue
        data_list = [temp_df[temp_df['group'] == grp][new_col].dropna() for grp in groups]
        pos_list = [center + offsets[g_idx] for g_idx in range(n_groups)]
        colors = [palette_cfg[grp]['color'] for grp in groups]
        plot_grouped_boxes_with_points(ax_d, data_list, pos_list, colors, widths=box_width * 0.85)

        p_val = compute_kruskal_p(temp_df, new_col)
        stars = get_significance_stars(p_val)
        if stars and stars != "ns":
            all_d = pd.concat([d for d in data_list if len(d) > 0])
            if len(all_d) > 0:
                ax_d.text(center, all_d.max() + (all_d.max() - all_d.min()) * 0.08, stars,
                          ha='center', va='bottom', fontweight='bold', color='crimson', fontsize=11)

    ax_d.set_xticks(range(len(dev_cols)))
    ax_d.set_xticklabels([item[2] for item in dev_cols], fontweight='semibold')

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette_cfg[grp]['color'], edgecolor='black', alpha=0.7)
        for grp in groups
    ]
    labels = [
        f"{palette_cfg[grp]['label']} (n={len(df[df['group'] == grp])})"
        for grp in groups
    ]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=n_groups, frameon=True, facecolor='#fafafa', edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = f"{output_prefix}_Figure3_Limb_Symmetry_Analysis.png"
    pdf_path = f"{output_prefix}_Figure3_Limb_Symmetry_Analysis.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# Figure 4: Global Feature Correlation & 2D Manifold Embedding
# ==============================================================================
def generate_figure4_correlation_and_pca(df, output_prefix, palette_cfg):
    """
    Panel A: Hierarchical Clustered Correlation Heatmap (Selected Representative Features)
    Panel B: 2D PCA Trajectory & Group Separation with 95% Confidence Ellipses & Feature Loadings.
    """
    fig, (ax_heat, ax_pca) = plt.subplots(1, 2, figsize=(16.5, 7.8), dpi=300,
                                           gridspec_kw={'width_ratios': [1.15, 1.0], 'wspace': 0.32})
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.18, top=0.91)

    candidate_features = [
        ('FL_stance_time_values_mean', 'FL Stance Time'),
        ('FR_stance_time_values_mean', 'FR Stance Time'),
        ('BL_stance_time_values_mean', 'BL Stance Time'),
        ('BR_stance_time_values_mean', 'BR Stance Time'),
        ('FL_stride_length_values_mean', 'FL Stride Length'),
        ('FR_stride_length_values_mean', 'FR Stride Length'),
        ('BL_stride_length_values_mean', 'BL Stride Length'),
        ('BR_stride_length_values_mean', 'BR Stride Length'),
        ('FL_duty_factor_values_mean', 'FL Duty Factor'),
        ('FR_duty_factor_values_mean', 'FR Duty Factor'),
        ('BL_duty_factor_values_mean', 'BL Duty Factor'),
        ('BR_duty_factor_values_mean', 'BR Duty Factor'),
        ('back_height_feature_H_values_mean', 'Back Height (H)'),
        ('front_hoof_release_angle_alpha1_values_mean', r'Front Hoof $\alpha_1$'),
        ('hind_hoof_release_angle_alpha1_values_mean', r'Hind Hoof $\alpha_1$'),
        ('back_neck_angle_beta1_values_mean', r'Posture $\beta_1$'),
        ('head_bobbing_head_y_amp_values_mean', 'Head Nod Amp'),
        ('symmetry_ratio_Left Front / Right Front_stance_time', 'LF/RF Stance Ratio')
    ]

    valid_feats = [item for item in candidate_features if item[0] in df.columns]
    feat_keys = [item[0] for item in valid_feats]
    feat_labels = [item[1] for item in valid_feats]

    sub_df = df[feat_keys].copy()
    for col in feat_keys:
        if sub_df[col].isnull().sum() > 0:
            sub_df[col] = sub_df[col].fillna(sub_df[col].median())

    corr_matrix = sub_df.corr(method='spearman').values
    n_feats = len(feat_keys)

    # Hierarchical clustering reordering
    dist_matrix = 1.0 - np.clip(corr_matrix, -1.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    order = leaves_list(linkage_matrix)

    ordered_corr = corr_matrix[order, :][:, order]
    ordered_labels = [feat_labels[i] for i in order]

    # Panel A: Heatmap
    ax_heat.set_title('(A) Clustered Feature Correlation Matrix',
                      fontweight='bold', pad=12, loc='left')

    im = ax_heat.imshow(ordered_corr, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='equal')
    ax_heat.set_xticks(range(n_feats))
    ax_heat.set_yticks(range(n_feats))
    ax_heat.set_xticklabels(ordered_labels, rotation=45, ha='right', fontsize=8.5)
    ax_heat.set_yticklabels(ordered_labels, fontsize=8.5)

    for i in range(n_feats):
        for j in range(n_feats):
            val = ordered_corr[i, j]
            if abs(val) >= 0.5:
                text_col = 'white' if abs(val) > 0.65 else 'black'
                ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center',
                             color=text_col, fontsize=6.5, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman Correlation Coefficient', fontsize=9.5)
    cbar.ax.tick_labelsize = 8.5

    # Panel B: PCA 2D Projection
    ax_pca.set_title('(B) Locomotion Feature Space (2D PCA Projection)',
                     fontweight='bold', pad=12, loc='left')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(sub_df)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_features)
    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0

    ax_pca.set_xlabel(f'Principal Component 1 ({var1:.1f}% Variance)', fontweight='semibold')
    ax_pca.set_ylabel(f'Principal Component 2 ({var2:.1f}% Variance)', fontweight='semibold')
    ax_pca.grid(True, linestyle='--', alpha=0.5, zorder=0)

    groups = list(palette_cfg.keys())
    for grp in groups:
        mask = (df['group'] == grp).values
        pts = pca_coords[mask]
        if len(pts) == 0:
            continue
        cfg = palette_cfg[grp]
        ax_pca.scatter(
            pts[:, 0], pts[:, 1],
            color=cfg['color'],
            label=f"{cfg['label']} (n={len(pts)})",
            edgecolors='black',
            linewidths=0.8,
            s=65,
            alpha=0.9,
            zorder=4
        )
        if len(pts) >= 4:
            draw_confidence_ellipse(
                pts[:, 0], pts[:, 1], ax_pca,
                n_std=1.96,
                facecolor=cfg['light'],
                edgecolor=cfg['color'],
                linewidth=1.5,
                linestyle='--',
                alpha=0.25,
                zorder=2
            )
        centroid = np.mean(pts, axis=0)
        ax_pca.scatter(
            centroid[0], centroid[1],
            marker='X', s=130, color=cfg['color'],
            edgecolors='black', linewidths=1.2, zorder=5
        )

    # Top PCA loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    feature_magnitudes = np.linalg.norm(loadings[:, :2], axis=1)
    top_arrow_indices = np.argsort(feature_magnitudes)[-4:]

    arrow_scale = np.max(np.abs(pca_coords)) * 0.75 / (np.max(feature_magnitudes) + 1e-6)
    for idx in top_arrow_indices:
        lx = loadings[idx, 0] * arrow_scale
        ly = loadings[idx, 1] * arrow_scale
        ax_pca.annotate(
            '', xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#2b2b2b', lw=1.3, alpha=0.75)
        )
        ax_pca.text(
            lx * 1.12, ly * 1.12, feat_labels[idx],
            fontsize=8, fontweight='bold', color='#1a1a1a',
            ha='center', va='center',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='#ffffff', edgecolor='none', alpha=0.7)
        )

    ax_pca.legend(loc='best', frameon=True, facecolor='#fafafa', edgecolor='#cccccc', fontsize=9.5)
    png_path = f"{output_prefix}_Figure4_Feature_Correlation_and_PCA.png"
    pdf_path = f"{output_prefix}_Figure4_Feature_Correlation_and_PCA.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# Table 1: Comprehensive Statistical Summary Table (Markdown & CSV)
# ==============================================================================
def generate_summary_table(df, output_prefix, dataset_type='3level'):
    """
    Compute Mean ± SD, Kruskal-Wallis / Mann-Whitney test, effect size,
    and output formatted Markdown & CSV summary tables.
    """
    feature_categories = {
        'Spatio-Temporal Features': [
            ('FL_stance_time_values_mean', 'FL Stance Time (s)'),
            ('FR_stance_time_values_mean', 'FR Stance Time (s)'),
            ('BL_stance_time_values_mean', 'BL Stance Time (s)'),
            ('BR_stance_time_values_mean', 'BR Stance Time (s)'),
            ('FL_stride_length_values_mean', 'FL Stride Length (px)'),
            ('FR_stride_length_values_mean', 'FR Stride Length (px)'),
            ('BL_stride_length_values_mean', 'BL Stride Length (px)'),
            ('BR_stride_length_values_mean', 'BR Stride Length (px)'),
            ('FL_stride_time_values_mean', 'FL Stride Time (s)'),
            ('FR_stride_time_values_mean', 'FR Stride Time (s)'),
            ('BL_stride_time_values_mean', 'BL Stride Time (s)'),
            ('BR_stride_time_values_mean', 'BR Stride Time (s)'),
            ('FL_duty_factor_values_mean', 'FL Duty Factor'),
            ('FR_duty_factor_values_mean', 'FR Duty Factor'),
            ('BL_duty_factor_values_mean', 'BL Duty Factor'),
            ('BR_duty_factor_values_mean', 'BR Duty Factor')
        ],
        'Posture & Angular Biomechanics': [
            ('back_height_feature_H_values_mean', 'Back Height Metric H (px)'),
            ('front_hoof_release_angle_alpha1_values_mean', 'Front Hoof Release Angle α1 (°)'),
            ('front_hoof_release_angle_alpha2_values_mean', 'Front Hoof Release Angle α2 (°)'),
            ('hind_hoof_release_angle_alpha1_values_mean', 'Hind Hoof Release Angle α1 (°)'),
            ('hind_hoof_release_angle_alpha2_values_mean', 'Hind Hoof Release Angle α2 (°)'),
            ('back_neck_angle_beta1_values_mean', 'Neck-Back Angle β1 (°)'),
            ('back_neck_angle_beta2_values_mean', 'Torso Angle β2 (°)'),
            ('head_bobbing_head_y_amp_values_mean', 'Head Bobbing Amplitude (px)'),
            ('head_bobbing_head_y_sd_values_mean', 'Head Bobbing SD (px)')
        ],
        'Limb Bilateral & Coordination Symmetry': [
            ('symmetry_ratio_Left Front / Right Front_stance_time', 'LF / RF Stance Time Ratio'),
            ('symmetry_ratio_Left Front / Right Front_stride_length', 'LF / RF Stride Length Ratio'),
            ('symmetry_ratio_Left Front / Right Front_stride_time', 'LF / RF Stride Time Ratio'),
            ('symmetry_ratio_Left Hind / Right Hind_stance_time', 'LH / RH Stance Time Ratio'),
            ('symmetry_ratio_Left Hind / Right Hind_stride_length', 'LH / RH Stride Length Ratio'),
            ('symmetry_ratio_Left Hind / Right Hind_stride_time', 'LH / RH Stride Time Ratio'),
            ('symmetry_ratio_Front / Hind_stance_time', 'Front / Hind Stance Time Ratio'),
            ('symmetry_ratio_Front / Hind_stride_length', 'Front / Hind Stride Length Ratio'),
            ('symmetry_ratio_Front / Hind_stride_time', 'Front / Hind Stride Time Ratio')
        ]
    }

    rows = []
    if dataset_type == '91videos_3level':
        groups = ['level0_sound', 'level1_medium', 'level2_severe']
    elif dataset_type == '3level':
        groups = ['level1_sound', 'level2_medium', 'level3_lame']
    else:
        groups = ['sound', 'lame']

    for cat_name, feats in feature_categories.items():
        for col_name, display_name in feats:
            if col_name not in df.columns:
                continue

            grp_stats = {}
            for grp in groups:
                series = df[df['group'] == grp][col_name].dropna()
                if len(series) > 0:
                    mean_val = series.mean()
                    std_val = series.std()
                    grp_stats[grp] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    grp_stats[grp] = "N/A"

            p_val = compute_kruskal_p(df, col_name)
            stars = get_significance_stars(p_val)

            valid_series = [df[df['group'] == grp][col_name].dropna().values for grp in groups]
            clean_groups = [g for g in valid_series if len(g) >= 2]
            n_total = sum(len(g) for g in clean_groups)
            h_stat = np.nan
            eff_size = np.nan

            if len(clean_groups) >= 2 and n_total > len(clean_groups):
                try:
                    h_stat, _ = stats.kruskal(*clean_groups)
                    eff_size = (h_stat - len(clean_groups) + 1) / (n_total - len(clean_groups))
                    eff_size = max(0.0, eff_size)
                except Exception:
                    pass

            row_entry = {
                'Category': cat_name,
                'Feature': display_name,
                'Raw_Key': col_name
            }
            for grp in groups:
                row_entry[grp] = grp_stats.get(grp, 'N/A')

            row_entry['H_stat'] = f"{h_stat:.2f}" if not np.isnan(h_stat) else "N/A"
            row_entry['p_value'] = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            row_entry['Significance'] = stars
            row_entry['Effect_Size_eta2'] = f"{eff_size:.3f}" if not np.isnan(eff_size) else "N/A"
            rows.append(row_entry)

    summary_df = pd.DataFrame(rows)

    csv_path = f"{output_prefix}_Table1_Gait_Features_Statistics.csv"
    summary_df.to_csv(csv_path, index=False)

    md_path = f"{output_prefix}_Table1_Gait_Features_Statistics.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Table 1: Comprehensive Statistical Summary of Gait Features ({dataset_type.upper()})\n\n")
        f.write("Values are presented as **Mean ± Standard Deviation**. Statistical comparisons were performed using Kruskal-Wallis test.\n")
        f.write("Significance: `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` not significant (p ≥ 0.05).\n\n")

        headers = ['Category', 'Gait Feature'] + groups + ['H-Stat', 'p-value', 'Sig.', 'Effect Size (η²)']
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join([':---'] * 2 + [':---:'] * (len(headers) - 2)) + " |\n")

        current_cat = ""
        for _, r in summary_df.iterrows():
            cat_display = r['Category'] if r['Category'] != current_cat else ""
            current_cat = r['Category']
            vals = [cat_display, r['Feature']]
            for grp in groups:
                vals.append(r[grp])
            vals.extend([r['H_stat'], r['p_value'], r['Significance'], r['Effect_Size_eta2']])
            f.write("| " + " | ".join(vals) + " |\n")

    print(f"Saved: {csv_path} and {md_path}")
    return summary_df


# ==============================================================================
# Main Runner
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Process 91-Video 3-Level Dataset (Level 0: Sound, Level 1: Medium, Level 2: Severe)
    json_91video = os.path.join(base_dir, '6-classified_3level_features.json')
    if os.path.exists(json_91video):
        print("\n=======================================================")
        print("Processing 91-Video 3-Level Dataset (Level 0, Level 1, Level 2)...")
        print("=======================================================")
        df_91video = load_dataset(json_91video, dataset_type='91videos_3level')
        prefix_91v = os.path.join(out_dir, '91videos')

        generate_figure1_spatiotemporal(df_91video, prefix_91v, PALETTE_91VIDEOS)
        generate_figure2_posture_kinematics(df_91video, prefix_91v, PALETTE_91VIDEOS)
        generate_figure3_symmetry(df_91video, prefix_91v, PALETTE_91VIDEOS)
        generate_figure4_correlation_and_pca(df_91video, prefix_91v, PALETTE_91VIDEOS)
        generate_summary_table(df_91video, prefix_91v, dataset_type='91videos_3level')

    # 2. Process 39-Video 3-Level Dataset (Pressure Mat Ground Truth)
    json_3level = os.path.join(base_dir, '6-classified_lame_level_features.json')
    if os.path.exists(json_3level):
        print("\n=======================================================")
        print("Processing 39-Video 3-Level Dataset (Pressure Mat)...")
        print("=======================================================")
        df_3level = load_dataset(json_3level, dataset_type='3level')
        prefix_3l = os.path.join(out_dir, '3levels')

        generate_figure1_spatiotemporal(df_3level, prefix_3l, PALETTE_3LEVELS)
        generate_figure2_posture_kinematics(df_3level, prefix_3l, PALETTE_3LEVELS)
        generate_figure3_symmetry(df_3level, prefix_3l, PALETTE_3LEVELS)
        generate_figure4_correlation_and_pca(df_3level, prefix_3l, PALETTE_3LEVELS)
        generate_summary_table(df_3level, prefix_3l, dataset_type='3level')

    # 2. Process 2-Level Dataset (Binary: Sound vs Lame)
    json_2level = os.path.join(base_dir, '6-classified_gait_features.json')
    if os.path.exists(json_2level):
        print("\n=======================================================")
        print("Processing 2-Level Gait Dataset (Sound vs Lame)...")
        print("=======================================================")
        df_2level = load_dataset(json_2level, dataset_type='2level')
        prefix_2l = os.path.join(out_dir, '2levels')

        generate_figure1_spatiotemporal(df_2level, prefix_2l, PALETTE_2LEVELS)
        generate_figure2_posture_kinematics(df_2level, prefix_2l, PALETTE_2LEVELS)
        generate_figure3_symmetry(df_2level, prefix_2l, PALETTE_2LEVELS)
        generate_figure4_correlation_and_pca(df_2level, prefix_2l, PALETTE_2LEVELS)
        generate_summary_table(df_2level, prefix_2l, dataset_type='2level')

    print("\nAll publication figures and statistical tables successfully generated in:")
    print(out_dir)


if __name__ == '__main__':
    main()
