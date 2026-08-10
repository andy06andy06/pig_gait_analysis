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
            key = vid if vid in data else (f"{vid}D" if f"{vid}D" in data else None)
            if key:
                feats = get_features_from_dict(data[key])
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

def evaluate_single_pair(t1, t2, scores_map, features_data):
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
    
    # Requirement: All 3 classes must have at least 6 samples for balanced evaluation
    if c0 < 6 or c1 < 6 or c2 < 6:
        return None
        
    X, y, _ = prepare_dataset_from_mapping(class_ids, features_data)
    if len(X) == 0:
        return None
        
    pipe_a = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=5)),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced'))
    ])
    
    pipe_b = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(score_func=f_classif, k=10)),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
    ])
    
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
    
    return {
        "t1": t1,
        "t2": t2,
        "counts": (c0, c1, c2),
        "acc_a": acc_a,
        "f1_a": f1_a,
        "acc_b": acc_b,
        "f1_b": f1_b,
    }

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
        
        score_lf_rf = max(lf_rf, 1/lf_rf) if lf_rf else 1.0
        score_lh_rh = max(lh_rh, 1/lh_rh) if lh_rh else 1.0
        score = max(score_lf_rf, score_lh_rh)
        scores_map[vid] = score
        
    features_path = os.path.join(base_dir, '3-standardized_keyframe_features.json')
    with open(features_path) as f:
        features_data = json.load(f)
        
    t1_values = np.arange(1.15, 1.45, 0.01)
    t2_values = np.arange(1.30, 1.65, 0.01)
    
    pairs = [(round(t1, 3), round(t2, 3)) for t1 in t1_values for t2 in t2_values if t2 > t1 + 0.04]
    
    print(f"Evaluating {len(pairs)} pairs in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_single_pair)(t1, t2, scores_map, features_data) for t1, t2 in pairs
    )
    
    results = [r for r in results if r is not None]
    print(f"Found {len(results)} valid configurations matching criteria.")
    results.sort(key=lambda x: max(x['acc_a'], x['acc_b']), reverse=True)
    
    print("\nTop 15 configurations sorted by Max(Acc_A, Acc_B):")
    print(f"{'t1':<6} | {'t2':<6} | {'Counts (L1,L2,L3)':<18} | {'Acc A':<6} | {'F1 A':<6} | {'Acc B':<6} | {'F1 B':<6}")
    print("-" * 75)
    for r in results[:15]:
        counts_str = f"({r['counts'][0]}, {r['counts'][1]}, {r['counts'][2]})"
        print(f"{r['t1']:<6.3f} | {r['t2']:<6.3f} | {counts_str:<18} | {r['acc_a']:<6.3f} | {r['f1_a']:<6.3f} | {r['acc_b']:<6.3f} | {r['f1_b']:<6.3f}")

if __name__ == '__main__':
    main()
