#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score

# Reuse functions or replicate them from code/6-classification.py
EXCLUDE_SYMMETRY_RATIO_FEATURES = True

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
                features[f"{prefix}{k}_median"] = np.median(v)
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = v
    return features

def prepare_dataset_from_mapping(class_ids, data):
    X = []
    y = []
    class_names = ["level1_sound", "level2_medium", "level3_lame"]
    
    for label, class_name in enumerate(class_names):
        for vid in class_ids[class_name]:
            if vid in data:
                feats = get_features_from_dict(data[vid])
                X.append(feats)
                y.append(label)
            elif f"{vid}D" in data:
                feats = get_features_from_dict(data[f"{vid}D"])
                X.append(feats)
                y.append(label)
                
    if not X:
        return np.array([]), np.array([]), []

    all_keys = sorted(list(set().union(*(d.keys() for d in X))))
    X_tmp = []
    for d in X:
        X_tmp.append([d.get(k, np.nan) for k in all_keys])

    X_tmp = np.array(X_tmp, dtype=float)
    feature_means = np.nanmean(X_tmp, axis=0)
    X_vec = np.where(np.isnan(X_tmp), feature_means, X_tmp)
    
    return X_vec, np.array(y), all_keys

def evaluate_thresholds(t1, t2, scores_map, features_data):
    # Classify based on thresholds
    class_ids = {"level1_sound": [], "level2_medium": [], "level3_lame": []}
    for vid, score in scores_map.items():
        if score < t1:
            class_ids["level1_sound"].append(vid)
        elif score < t2:
            class_ids["level2_medium"].append(vid)
        else:
            class_ids["level3_lame"].append(vid)
            
    c0 = len(class_ids["level1_sound"])
    c1 = len(class_ids["level2_medium"])
    c2 = len(class_ids["level3_lame"])
    
    # Conditions:
    # 1. Level 0 and Level 2 are the most populated
    # 2. Level 1 is the minority (specifically, count(level1) < count(level0) and count(level1) < count(level2))
    # 3. All classes must have at least some samples (e.g., Level 1 >= 2, Level 2 >= 4)
    if not (c0 > c1 and c2 > c1 and c1 >= 2 and c2 >= 4 and c0 >= 4):
        return None
        
    X, y, _ = prepare_dataset_from_mapping(class_ids, features_data)
    if len(X) == 0:
        return None
        
    # Evaluate using standard SVM pipeline
    # We will test two standard configs to see which one works better:
    # Config A: select k=5, linear kernel
    # Config B: select k='all', RBF kernel
    pipe_a = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=5)),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced'))
    ])
    
    pipe_b = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
    ])
    
    # Evaluate using LOO CV
    loo = LeaveOneOut()
    y_true, y_pred_a, y_pred_b = [], [], []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_true.append(y_test[0])
        
        try:
            pipe_a.fit(X_train, y_train)
            y_pred_a.append(pipe_a.predict(X_test)[0])
        except Exception:
            y_pred_a.append(-1)
            
        try:
            pipe_b.fit(X_train, y_train)
            y_pred_b.append(pipe_b.predict(X_test)[0])
        except Exception:
            y_pred_b.append(-1)
            
    acc_a = accuracy_score(y_true, y_pred_a)
    f1_a = f1_score(y_true, y_pred_a, average='macro', zero_division=0)
    
    acc_b = accuracy_score(y_true, y_pred_b)
    f1_b = f1_score(y_true, y_pred_b, average='macro', zero_division=0)
    
    # Check if level3_lame recall is non-zero in at least one config
    # level3 corresponds to class index 2
    y_true_arr = np.array(y_true)
    y_pred_a_arr = np.array(y_pred_a)
    y_pred_b_arr = np.array(y_pred_b)
    
    recall_c2_a = np.sum((y_true_arr == 2) & (y_pred_a_arr == 2)) / np.sum(y_true_arr == 2) if np.sum(y_true_arr == 2) > 0 else 0.0
    recall_c2_b = np.sum((y_true_arr == 2) & (y_pred_b_arr == 2)) / np.sum(y_true_arr == 2) if np.sum(y_true_arr == 2) > 0 else 0.0
    
    return {
        "t1": t1,
        "t2": t2,
        "counts": (c0, c1, c2),
        "acc_a": acc_a,
        "f1_a": f1_a,
        "recall_c2_a": recall_c2_a,
        "acc_b": acc_b,
        "f1_b": f1_b,
        "recall_c2_b": recall_c2_b,
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, '../videos')
    
    # 1. Parse pressuremat files
    scores_map = {}
    for p in sorted(glob.glob(os.path.join(videos_dir, '*_pressuremat.json'))):
        vid = os.path.basename(p).replace('_pressuremat.json', '')
        with open(p) as f:
            data = json.load(f)
        sections = data.get('symmetry_table', {}).get('sections', {})
        lf_rf = sections.get('Left Front / Right Front', {}).get('Max Force')
        lh_rh = sections.get('Left Hind / Right Hind', {}).get('Max Force')
        
        score_lf_rf = max(lf_rf, 1/lf_rf) if lf_rf else 1.0
        score_lh_rh = max(lh_rh, 1/lh_rh) if lh_rh else 1.0
        score = max(score_lf_rf, score_lh_rh)
        scores_map[vid] = score
        
    # 2. Load standardized features
    features_path = os.path.join(base_dir, '3-standardized_keyframe_features.json')
    with open(features_path) as f:
        features_data = json.load(f)
        
    # 3. Grid search over thresholds
    results = []
    # Generate grids of thresholds with step 0.01
    t1_values = np.arange(1.15, 1.45, 0.01)
    t2_values = np.arange(1.30, 1.65, 0.01)
    
    for t1 in t1_values:
        for t2 in t2_values:
            if t2 <= t1 + 0.04:
                continue
            res = evaluate_thresholds(round(t1, 3), round(t2, 3), scores_map, features_data)
            if res:
                results.append(res)
                
    # Sort results
    print(f"Found {len(results)} valid configurations matching criteria.")
    
    # Display top 15 by F1 score or recall
    results.sort(key=lambda x: max(x['f1_a'], x['f1_b']), reverse=True)
    
    print("\nTop 15 configurations sorted by Max(F1_A, F1_B):")
    print(f"{'t1':<6} | {'t2':<6} | {'Counts (L0,L1,L2)':<18} | {'Acc A':<5} | {'F1 A':<5} | {'Recall L2 A':<11} | {'Acc B':<5} | {'F1 B':<5} | {'Recall L2 B':<11}")
    print("-" * 105)
    for r in results[:15]:
        counts_str = f"({r['counts'][0]}, {r['counts'][1]}, {r['counts'][2]})"
        print(f"{r['t1']:<6.3f} | {r['t2']:<6.3f} | {counts_str:<18} | {r['acc_a']:<5.3f} | {r['f1_a']:<5.3f} | {r['recall_c2_a']:<11.3f} | {r['acc_b']:<5.3f} | {r['f1_b']:<5.3f} | {r['recall_c2_b']:<11.3f}")

if __name__ == '__main__':
    main()
