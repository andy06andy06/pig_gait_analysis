# Pig Gait Analysis Pipeline

This repository contains a comprehensive pipeline for analyzing pig locomotion using DeepLabCut (DLC) pose estimation data. The workflow includes trajectory cleaning, keyframe detection, kinematic feature extraction, unsupervised clustering, supervised lameness classification, and ground-truth validation against pressure mat data.

## 📂 Directory Structure

The scripts rely on a specific directory structure. Ensure your project is organized as follows:

```text
root/
├── videos/                        # INPUT: Raw Data
│   ├── *.h5                       # DeepLabCut output files (filtered)
│   ├── *.mp4                      # Original video files
│   ├── *_pressuremat.json         # (Optional) Ground truth pressure mat data
│   ├── plots/                     # OUTPUT: Trajectory visualization
│   └── keyframe_image/            # OUTPUT: Keyframe snapshots with overlays
│
└── code/                          # SCRIPTS
    ├── 1-positioning.py
    ├── 2-keyframe_coords.py
    ├── 3-keyframe_features.py
    ├── 4-pressuremat_vs_video.py
    ├── 5-feature_analysis.py
    ├── 6-classification.py
    ├── u-convert_pressuremat.py
    ├── u-loss_per_keypoint.py
    └── *.json                     # Intermediate data files are generated here
```

-----

## 🚀 Pipeline Workflow

Run the scripts in the numerical order below to process raw DLC data into analyzed gait features.

### Phase 1: Preprocessing & coordinate Extraction

**1. `1-positioning.py`**

  * **Purpose:** Cleans DLC trajectories by removing outliers, filling gaps, and detecting stance/swing phases based on hoof velocity.
  * **Input:** `../videos/*.h5`
  * **Output:** \* `1-keyframes_starts_only.json`: Frame indices where key events (hoof strikes) occur.
      * `1-keyframes_segments.json`: Detailed segment data (start/end frames, duration).
      * `../videos/plots/`: X-coordinate trajectory plots vs. detected phases.

**2. `2-keyframe_coords.py`**

  * **Purpose:** Extracts coordinate data and likelihoods specifically at the detected keyframes. Determines walking direction (Left/Right) and exports visualization images.
  * **Input:** `1-keyframes_starts_only.json`, `.h5` files, `.mp4` files.
  * **Output:** \* `2-keyframe_coords.json`: Structured coordinates for every keyframe.
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
  * **Output:** `3-keyframe_feature.json` (The master dataset).

### Phase 3: Analysis & Machine Learning

**4. `5-feature_analysis.py`**

  * **Purpose:** Performs unsupervised analysis to find patterns in the gait data.
  * **Methods:** Z-score standardization, K-Means clustering, PCA/t-SNE projection, and Pearson correlation matrices.
  * **Input:** `3-keyframe_feature.json`.
  * **Output:** \* `3-standardized_keyframe_features.json`
      * `5-kmeans_clusters.json`
      * `feature_analysis/`: Contains Correlation matrices, Cluster visualizations (PCA/t-SNE), and Feature deviation bar charts.

**5. `6-classification.py`**

  * **Purpose:** Performs supervised classification to detect "Lame" vs. "Sound" pigs.
  * **Methods:** Support Vector Machine (SVM) with RBF kernel using Leave-One-Out Cross-Validation (LOOCV).
  * **Input:** `gait_features.json` (ensure this matches the output from step 3 or is renamed).
  * **Output:** \* `classification/svm_confusion_matrix.png`
      * Console report: Accuracy score and Classification report.

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
  * **Input:** `3-keyframe_feature.json` and `*_pressuremat.json`.
  * **Output:** \* `pressuremat_vs_video/`: Scatter plots, Bland-Altman plots, and `validation_summary.txt`.

-----

## 🛠️ Utilities

**`u-loss_per_keypoint.py`**

  * **Purpose:** Analyzes the DeepLabCut training CSV to visualize model performance per body part.
  * **Input:** DLC `keypoint-results.csv`.
  * **Output:** Bar charts showing Train vs. Test error (in pixels) for every keypoint.