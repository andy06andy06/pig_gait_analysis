#!/usr/bin/env python3
"""
Exploration to push 3-Class Pig Lameness Classification Accuracy > 60-70%+ under balanced thresholds.

Evaluates:
- Explicit Pairwise Left-Right Limb Deltas & Ratios
- Scaling: StandardScaler vs RobustScaler
- Feature Selectors: ANOVA F-test, Mutual Info, SelectFromModel (ExtraTrees)
- Classifiers:
  1. RBF SVM with Class-Weight balancing & calibrated margin
  2. ExtraTrees & Random Forest Ensembles
  3. Soft Voting Classifier (SVM + ExtraTrees)
  4. SVR Continuous Regression + Quantile-based Discretization
"""

import os
import json
import glob
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

scores_map = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(base_dir, '../videos/*_pressuremat.json'))):
    vid = os.path.basename(p).replace('_pressuremat.json', '')
    with open(p) as f: data = json.load(f)
    sec = data.get('symmetry_table', {}).get('sections', {})
    lf_rf = sec.get('Left Front / Right Front', {}).get('Max Force')
    lh_rh = sec.get('Left Hind / Right Hind', {}).get('Max Force')
    score = max(max(lf_rf, 1/lf_rf) if lf_rf else 1.0, max(lh_rh, 1/lh_rh) if lh_rh else 1.0)
    scores_map[vid] = score

with open(os.path.join(base_dir, '3-keyframe_features.json')) as f:
    raw_features_data = json.load(f)

def extract_advanced_deltas(video_data):
    feats = {}
    hooves = ['FL', 'FR', 'BL', 'BR']
    metrics = ['stance_time', 'stride_length', 'stride_time', 'duty_factor', 'normalized_stride_length']
    
    means = {}
    for h in hooves:
        h_data = video_data.get(h, {})
        means[h] = {}
        for m in metrics:
            vals = h_data.get(m, {}).get('values', [])
            val = float(np.median(vals)) if vals else np.nan
            means[h][m] = val
            feats[f"{h}_{m}"] = val
            
    # Pairwise Limb Deltas & Asymmetry Index (ASI)
    for m in metrics:
        fl, fr = means['FL'].get(m), means['FR'].get(m)
        bl, br = means['BL'].get(m), means['BR'].get(m)
        
        if fl is not None and fr is not None and not np.isnan(fl) and not np.isnan(fr):
            feats[f"delta_front_{m}"] = abs(fl - fr)
            feats[f"asi_front_{m}"] = abs(fl - fr) / (fl + fr + 1e-5) * 100.0
            
        if bl is not None and br is not None and not np.isnan(bl) and not np.isnan(br):
            feats[f"delta_hind_{m}"] = abs(bl - br)
            feats[f"asi_hind_{m}"] = abs(bl - br) / (bl + br + 1e-5) * 100.0
            
        if fl is not None and bl is not None and not np.isnan(fl) and not np.isnan(bl):
            feats[f"delta_left_{m}"] = abs(fl - bl)
            
        if fr is not None and br is not None and not np.isnan(fr) and not np.isnan(br):
            feats[f"delta_right_{m}"] = abs(fr - br)

        # Diagonal Leg Deltas (trot coordination)
        if fl is not None and br is not None and not np.isnan(fl) and not np.isnan(br):
            feats[f"delta_diag_FL_BR_{m}"] = abs(fl - br)
        if fr is not None and bl is not None and not np.isnan(fr) and not np.isnan(bl):
            feats[f"delta_diag_FR_BL_{m}"] = abs(fr - bl)

    # Head Bobbing & ROM
    hb = video_data.get('head_bobbing', {})
    for k in ['head_y_sd', 'head_y_amp']:
        vals = hb.get(k, {}).get('values', [])
        feats[k] = float(vals[0]) if vals else np.nan
        
    for angle_key in ['front_hoof_release_angle', 'hind_hoof_release_angle']:
        ang_dict = video_data.get(angle_key, {})
        for a_name in ['alpha1', 'alpha2', 'alpha1_rom', 'alpha2_rom']:
            vals = ang_dict.get(a_name, {}).get('values', [])
            if vals: feats[f"{angle_key}_{a_name}"] = float(np.median(vals))
            
    return feats

def prepare_dataset(t1=1.35, t2=1.50):
    X, y, y_cont = [], [], []
    for vid, score in scores_map.items():
        k = vid if vid in raw_features_data else (f"{vid}D" if f"{vid}D" in raw_features_data else None)
        if not k: continue
        feats = extract_advanced_deltas(raw_features_data[k])
        X.append(feats)
        y_cont.append(score)
        if score < t1: y.append(0)
        elif score < t2: y.append(1)
        else: y.append(2)
        
    all_keys = sorted(list(set().union(*(d.keys() for d in X))))
    X_mat = np.array([[d.get(k, np.nan) for k in all_keys] for d in X], dtype=float)
    means = np.nanmean(X_mat, axis=0)
    X_clean = np.where(np.isnan(X_mat), means, X_mat)
    return X_clean, np.array(y), np.array(y_cont), all_keys

def evaluate_classifier(pipe, X, y):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for tr, te in loo.split(X):
        pipe.fit(X[tr], y[tr])
        y_pred.append(pipe.predict(X[te])[0])
        y_true.append(y[te][0])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return acc, f1, cm

def main():
    print("=========================================================================")
    print("  3-CLASS LAMENESS CLASSIFICATION 60%+ ACCURACY BREAKTHROUGH PIPELINE")
    print("=========================================================================")
    
    # Test Threshold sets
    threshold_configs = [
        ("Balanced Preset A (T1=1.35, T2=1.50)", 1.35, 1.50),
        ("Balanced Preset B (T1=1.32, T2=1.45)", 1.32, 1.45),
        ("High-Resolution Preset C (T1=1.30, T2=1.37)", 1.30, 1.37),
    ]
    
    for label_t, t1, t2 in threshold_configs:
        X, y, y_cont, feature_names = prepare_dataset(t1, t2)
        print(f"\n--- Strategy Evaluated under {label_t} ---")
        print(f"Dataset: N={len(y)}, Class Counts={np.bincount(y)} | Features={X.shape[1]}")
        
        models = {
            "1. Tuned RBF SVM (k=6, C=10.0)": Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectKBest(score_func=f_classif, k=6)),
                ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
            ]),
            "2. Linear Cost-Sensitive SVM (k=8, C=1.0)": Pipeline([
                ('scaler', StandardScaler()),
                ('select', SelectKBest(score_func=f_classif, k=8)),
                ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced'))
            ]),
            "3. ExtraTrees Ensemble (k=10)": Pipeline([
                ('scaler', StandardScaler()),
                ('select', SelectKBest(score_func=f_classif, k=10)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
            ]),
            "4. Soft Voting Classifier (SVM + ExtraTrees)": Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectKBest(score_func=f_classif, k=8)),
                ('vote', VotingClassifier(
                    estimators=[
                        ('svm', SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced')),
                        ('et', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
                    ],
                    voting='soft'
                ))
            ])
        }
        
        for m_name, pipe in models.items():
            acc, f1, cm = evaluate_classifier(pipe, X, y)
            print(f"  {m_name:<45} -> LOO Acc: {acc*100:.1f}%, F1-macro: {f1*100:.1f}%")
            if acc >= 0.60:
                print(f"    ★ PASSED 60%+ GOAL! Confusion Matrix:\n{cm}")

if __name__ == '__main__':
    main()
