# Pig Gait Analysis Pipeline

This repository contains a comprehensive pipeline for analyzing pig locomotion using DeepLabCut (DLC) pose estimation data. The workflow includes trajectory cleaning, keyframe detection, kinematic feature extraction, unsupervised clustering, supervised lameness classification, and ground-truth validation against pressure mat data.

## 📂 Directory Structure

The scripts rely on a specific directory structure. Ensure your project is organized as follows:

```text
root/
├── videos/                                # INPUT: Raw Data
│   ├── *.h5                               # DeepLabCut output files (filtered, shuffle10)
│   ├── *.mp4                              # Original video files (91 total)
│   ├── *_pressuremat.json                 # Ground truth pressure mat data (39 paired)
│   ├── plots/                             # OUTPUT: Trajectory visualization
│   ├── keyframe_image/                    # OUTPUT: Keyframe snapshots with overlays
│   └── classified_video_3levels/          # DATASET: 91-Video 3-Level Dataset (Level 0, 1, 2)
│       ├── level0_sound/                  # 77 videos (Sound / 健康)
│       ├── level1_medium/                 # 12 videos (Medium Lameness / 輕中度跛行)
│       ├── level2_severe/                 # 2 videos (Severe Lameness / 最嚴重跛行)
│       ├── classification_summary.json    # Full dataset annotation index
│       └── classification_summary.csv
│
├── archive_deprecated_datasets/           # ARCHIVE: Old binary & exploratory datasets
│
└── code/                                  # SCRIPTS
    ├── 1-positioning.py
    ├── 2-keyframe_coords.py
    ├── 3-keyframe_features.py
    ├── 4-pressuremat_vs_video.py
    ├── 5-feature_analysis.py
    ├── 6-classification.py
    ├── generate_paper_gait_plots.py
    ├── u-create_3level_dataset.py
    └── *.json                             # Master & intermediate feature matrices
```

-----

## 🚀 Pipeline Workflow

Run the scripts in the numerical order below to process raw DLC data into analyzed gait features.

### Phase 1: Preprocessing & coordinate Extraction

**1. `1-positioning.py`**

  * **Purpose:** Cleans DLC trajectories by removing outliers, filling gaps, and detecting stance/swing phases based on hoof velocity.
  * **Input:** `../videos/*.h5`
  * **Output:** 
      * `1-keyframes_starts_only.json`: Frame indices where key events (hoof strikes) occur.
      * `1-keyframes_segments.json`: Detailed segment data (start/end frames, duration).
      * `../videos/plots/`: X-coordinate trajectory plots vs. detected phases.

**2. `2-keyframe_coords.py`**

  * **Purpose:** Extracts coordinate data and likelihoods specifically at the detected keyframes. Determines walking direction (Left/Right) and exports visualization images.
  * **Input:** `1-keyframes_starts_only.json`, `.h5` files, `.mp4` files.
  * **Output:** 
      * `2-keyframe_coords.json`: Structured coordinates for every keyframe.
      * `../videos/keyframe_image/`: Cropped images of the pigs with skeleton overlays.

### Phase 2: Feature Engineering

**3. `3-keyframe_features.py`**

  * **Purpose:** Calculates specific kinematic gait parameters from the coordinates.
  * **Computed Features:**
      * **Temporal:** Stride time, Stance time.
      * **Spatial:** Stride length (pixels), Back height difference ($H$).
      * **Angular:** Hoof release angles ($\alpha_1, \alpha_2$), Back/Neck angles ($\beta_1, \beta_2$).
      * **Symmetry:** Ratios (Left/Right, Front/Hind).
  * **Input:** `2-keyframe_coords.json`, `1-keyframes_starts_only.json`.
  * **Output:** `3-keyframe_features.json` (The master dataset).

### Phase 3: Analysis & Machine Learning

**4. `5-feature_analysis.py`**

  * **Purpose:** Performs unsupervised analysis to find patterns in the gait data.
  * **Methods:** Z-score standardization, K-Means clustering, PCA/t-SNE projection, and Pearson correlation matrices.
  * **Input:** `3-keyframe_features.json`.
  * **Output:** 
      * `3-standardized_keyframe_features.json`
      * `5-kmeans_clusters.json`
      * `feature_analysis/`: Contains Correlation matrices, Cluster visualizations (PCA/t-SNE), and Feature deviation bar charts.

**5. `6-classification.py`**

  * **Purpose:** Performs 3-level supervised lameness classification (Level 0: Sound, Level 1: Medium, Level 2: Severe) using the 91-video dataset.
  * **Methods:** Ordinal Support Vector Machine (SVM) with Oversampling, Feature Selection (`SelectKBest`), Grid Search, and Leave-One-Out Cross-Validation (LOO CV), plus Permutation Testing and Strict Nested LOO CV.
  * **Input:** `videos/classified_video_3levels/`, `3-standardized_keyframe_features.json` / `3-keyframe_features.json`.
  * **Output:** 
      * `6-classified_3level_features.json`: Master 3-level features for all 91 pigs.
      * `classification_3levels/`: Contains Confusion Matrix, 2D Decision Boundary plots, LOO results, and Nested CV summaries.

**6. `generate_paper_gait_plots.py`**

  * **Purpose:** Generates publication-ready figures and statistical summary tables across all 3 lameness levels.
  * **Methods:** Non-parametric ANOVA (Kruskal-Wallis), pairwise Mann-Whitney U tests with FDR correction, PCA ellipses, hierarchical clustering heatmaps.
  * **Input:** `6-classified_3level_features.json`.
  * **Output:** 
      * `paper_figures/`: Figures 1–4 (PNG + PDF) and Table 1 (`.csv` + `.md`) for the 91-video dataset.

-----

## ⚖️ Validation (Pressure Mat)

Use these scripts to validate video-derived metrics against ground-truth pressure mat data.

**A. `u-convert_pressuremat.py`**

  * **Purpose:** Converts raw tab-delimited pressure mat text files into usable JSON format.
  * **Input:** Raw pressure mat files in `../videos/`.
  * **Output:** `*_pressuremat.json`.

**B. `4-pressuremat_vs_video.py`**

  * **Purpose:** Matches video trials with pressure mat trials to quantify accuracy.
  * **Statistics:** Pearson/Spearman correlation, Bland-Altman agreement (Bias & Limits of Agreement), and ICC(2,1).
  * **Input:** `3-keyframe_features.json` and `*_pressuremat.json`.
  * **Output:** 
      * `pressuremat_vs_video/`: Scatter plots, Bland-Altman plots, and `validation_summary.txt`.

-----

## 🛠️ Utilities

**`u-loss_per_keypoint.py`**

  * **Purpose:** Analyzes the DeepLabCut training CSV to visualize model performance per body part.
  * **Input:** DLC `keypoint-results.csv`.
  * **Output:** Bar charts showing Train vs. Test error (in pixels) for every keypoint.
