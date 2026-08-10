#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score

EXCLUDE_SYMMETRY_RATIO_FEATURES = False

def get_features_from_dict(d, prefix=''):
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']:
            continue
        if EXCLUDE_SYMMETRY_RATIO_FEATURES and 'symmetry_ratio' in f"{prefix}{k}":
            continue
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_median"] = float(np.median(v))
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = float(v)
    return features

def prepare_dataset_for_scores(scores_map, t1, t2, features_data, is_binary=False):
    X, y = [], []
    for vid, score in scores_map.items():
        key = vid if vid in features_data else (f"{vid}D" if f"{vid}D" in features_data else None)
        if not key:
            continue
        feats = get_features_from_dict(features_data[key])
        
        if is_binary:
            if score < t1:
                y.append(0) # Sound
                X.append(feats)
            elif score >= t2:
                y.append(1) # Lame
                X.append(feats)
        else:
            if score < t1: label = 0
            elif score < t2: label = 1
            else: label = 2
            y.append(label)
            X.append(feats)
            
    if not X:
        return np.array([]), np.array([])
    all_keys = sorted(list(set().union(*(d.keys() for d in X))))
    X_mat = np.array([[d.get(k, np.nan) for k in all_keys] for d in X], dtype=float)
    means = np.nanmean(X_mat, axis=0)
    X_clean = np.where(np.isnan(X_mat), means, X_mat)
    return X_clean, np.array(y)

def eval_config(X, y, k=5, kernel='linear', C=1.0, scaler_type='standard'):
    if len(np.unique(y)) < 2:
        return 0.0, 0.0
    sc = RobustScaler() if scaler_type == 'robust' else StandardScaler()
    pipe = Pipeline([
        ('scaler', sc),
        ('select', SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))),
        ('svm', SVC(kernel=kernel, C=C, class_weight='balanced'))
    ])
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for tr, te in loo.split(X):
        pipe.fit(X[tr], y[tr])
        y_pred.append(pipe.predict(X[te])[0])
        y_true.append(y[te][0])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1

def eval_pair(t1, t2, scores_map, features_data):
    X, y = prepare_dataset_for_scores(scores_map, t1=t1, t2=t2, features_data=features_data, is_binary=False)
    if len(y) < 25 or len(np.unique(y)) < 3:
        return None
    acc1, f1_1 = eval_config(X, y, k=5, kernel='linear', C=0.1)
    acc2, f1_2 = eval_config(X, y, k=8, kernel='rbf', C=10.0, scaler_type='robust')
    acc3, f1_3 = eval_config(X, y, k=10, kernel='linear', C=1.0)
    max_acc = max(acc1, acc2, acc3)
    return (t1, t2, np.bincount(y), acc1, acc2, acc3, max_acc)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, '../videos')
    
    scores_map = {}
    for p in sorted(glob.glob(os.path.join(videos_dir, '*_pressuremat.json'))):
        vid = os.path.basename(p).replace('_pressuremat.json', '')
        with open(p) as f:
            data = json.load(f)
        sections = data.get('symmetry_table', {}).get('sections', {})
        lf_rf = sections.get('Left Front / Right Front', {}).get('Max Force')
        lh_rh = sections.get('Left Hind / Right Hind', {}).get('Max Force')
        score = max(max(lf_rf, 1/lf_rf) if lf_rf else 1.0, max(lh_rh, 1/lh_rh) if lh_rh else 1.0)
        scores_map[vid] = score
        
    features_path = os.path.join(base_dir, '3-standardized_keyframe_features.json')
    with open(features_path) as f:
        features_data = json.load(f)

    # --- 1. Search Binary Threshold (Sound vs Lame) ---
    print("=========================================================")
    print("  Binary Classification Threshold Search (Sound vs Lame)")
    print("=========================================================")
    best_bin_acc = 0.0
    best_bin_t = None
    
    for t in np.arange(1.15, 1.50, 0.01):
        X, y = prepare_dataset_for_scores(scores_map, t1=t, t2=t+0.10, features_data=features_data, is_binary=True)
        if len(y) < 15:
            continue
        acc, f1 = eval_config(X, y, k=5, kernel='linear', C=1.0)
        if acc > best_bin_acc:
            best_bin_acc = acc
            best_bin_t = (t, t+0.10, len(y), np.bincount(y))
            
    print(f"Best Binary Threshold t1={best_bin_t[0]:.2f}, t2={best_bin_t[1]:.2f} | Samples={best_bin_t[2]} {best_bin_t[3]} -> LOO Accuracy: {best_bin_acc*100:.1f}%")

    # --- 2. Search 3-Class Thresholds (Sound vs Medium vs Lame) ---
    print("\n=========================================================")
    print("  3-Class Threshold Search (Sound vs Medium vs Lame)")
    print("=========================================================")
    
    t1_values = np.arange(1.15, 1.45, 0.01)
    t2_values = np.arange(1.30, 1.65, 0.01)
    pairs = [(round(t1, 2), round(t2, 2)) for t1 in t1_values for t2 in t2_values if t2 > t1 + 0.04]
    
    results = Parallel(n_jobs=-1)(
        delayed(eval_pair)(t1, t2, scores_map, features_data) for t1, t2 in pairs
    )
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[6], reverse=True)
    
    print("\nTop 10 3-Class Threshold Configurations:")
    print(f"{'t1':<6} | {'t2':<6} | {'Class Counts (L1,L2,L3)':<22} | {'Max LOO Acc':<12}")
    print("-" * 65)
    for r in results[:10]:
        print(f"{r[0]:<6.2f} | {r[1]:<6.2f} | {str(r[2]):<22} | {r[6]*100:<12.1f}%")

if __name__ == '__main__':
    main()
