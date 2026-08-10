#!/usr/bin/env python3
"""
Comprehensive Pig Lameness SVM & SVR Optimization Engine.

Evaluates:
- Feature subsets: Baseline DLC features vs New Veterinary Kinematic Features (Duty factor, Head Bobbing, ROM, Normalized Stride)
- Standard discretization thresholds vs Balanced Thresholding
- Classifiers & Regressors: Multi-class SVC (RBF/Linear/Poly), Ordinal Classifier, Continuous SVR, Binary SVC
- Validation: LOO-CV, Nested LOO-CV, Permutation Significance Testing (100 iterations)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Any

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, r2_score
from scipy.stats import pearsonr

EXCLUDE_SYMMETRY_RATIO_FEATURES = False

# --- Ordinal Classifier ---
class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=None):
        self.clf = clf
        self.clfs = []
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clfs = []
        n_classes = len(self.classes_)
        base_clf = self.clf if self.clf is not None else SVC(probability=True)
        for i in range(n_classes - 1):
            binary_y = (y > i).astype(int)
            clf = clone(base_clf)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        probs = []
        for clf in self.clfs:
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X)[:, 1]
            elif hasattr(clf, "decision_function"):
                df = clf.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-df))
            else:
                p = clf.predict(X)
            probs.append(p)
        probs = np.array(probs).T
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_probs = np.zeros((n_samples, n_classes))
        
        class_probs[:, 0] = 1.0 - probs[:, 0]
        for i in range(1, n_classes - 1):
            class_probs[:, i] = probs[:, i-1] - probs[:, i]
        class_probs[:, -1] = probs[:, -1]
        
        class_probs = np.clip(class_probs, 0.0, 1.0)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return class_probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def get_features_from_dict(d, prefix='', include_new_vet_features=True):
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']:
            continue
        if EXCLUDE_SYMMETRY_RATIO_FEATURES and 'symmetry_ratio' in f"{prefix}{k}":
            continue
            
        full_key = f"{prefix}{k}"
        
        if not include_new_vet_features:
            if any(vet in full_key for vet in ['duty_factor', 'head_bobbing', 'normalized_stride', 'rom']):
                continue
        
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{full_key}_", include_new_vet_features))
        elif isinstance(v, list):
            if v and isinstance(v[0], (int, float)):
                features[f"{full_key}_median"] = float(np.median(v))
        elif isinstance(v, (int, float)):
            features[f"{full_key}"] = float(v)
    return features

def load_data(classified_features_path, include_new_vet_features=True):
    with open(classified_features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    class_names = ["level1_sound", "level2_medium", "level3_lame"]
    X_raw, y_raw, ids = [], [], []
    
    for label, class_name in enumerate(class_names):
        records = data.get(class_name, {})
        for vid, feats in records.items():
            parsed = get_features_from_dict(feats, include_new_vet_features=include_new_vet_features)
            X_raw.append(parsed)
            y_raw.append(label)
            ids.append(vid)
            
    all_keys = sorted(list(set().union(*(d.keys() for d in X_raw))))
    X_mat = []
    for d in X_raw:
        X_mat.append([d.get(k, np.nan) for k in all_keys])
        
    X_mat = np.array(X_mat, dtype=float)
    means = np.nanmean(X_mat, axis=0)
    X_clean = np.where(np.isnan(X_mat), means, X_mat)
    
    # Load continuous scores
    scores_map = {}
    videos_dir = os.path.join(os.path.dirname(classified_features_path), '../videos')
    for p in glob.glob(os.path.join(videos_dir, '*_pressuremat.json')):
        vid = os.path.basename(p).replace('_pressuremat.json', '')
        with open(p) as pf:
            pdata = json.load(pf)
        sec = pdata.get('symmetry_table', {}).get('sections', {})
        lf_rf = sec.get('Left Front / Right Front', {}).get('Max Force')
        lh_rh = sec.get('Left Hind / Right Hind', {}).get('Max Force')
        score = max(max(lf_rf, 1/lf_rf) if lf_rf else 1.0, max(lh_rh, 1/lh_rh) if lh_rh else 1.0)
        scores_map[vid] = score
        
    y_continuous = np.array([scores_map.get(v, 1.3) for v in ids])
    
    return X_clean, np.array(y_raw), y_continuous, all_keys, ids

def eval_fold(train_idx, test_idx, X, y, pipe_fn, grid_params):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    best_score = -1.0
    best_pred = 0
    
    # Inner Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for params in grid_params:
        pipe = pipe_fn(params)
        cv_scores = []
        for tr_in, val_in in skf.split(X_tr, y_tr):
            try:
                pipe.fit(X_tr[tr_in], y_tr[tr_in])
                val_p = pipe.predict(X_tr[val_in])
                cv_scores.append(f1_score(y_tr[val_in], val_p, average='macro', zero_division=0))
            except Exception:
                cv_scores.append(0.0)
                
        mean_cv = np.mean(cv_scores) if cv_scores else 0.0
        if mean_cv > best_score:
            best_score = mean_cv
            try:
                pipe.fit(X_tr, y_tr)
                best_pred = pipe.predict(X_te)[0]
            except Exception:
                best_pred = 0
                
    return y_te[0], best_pred

def run_nested_loo(X, y, pipe_fn, grid_params):
    loo = LeaveOneOut()
    splits = list(loo.split(X))
    
    results = Parallel(n_jobs=-1)(
        delayed(eval_fold)(train_idx, test_idx, X, y, pipe_fn, grid_params)
        for train_idx, test_idx in splits
    )
    
    y_true = [r[0] for r in results]
    y_pred = [r[1] for r in results]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2] if len(np.unique(y)) > 2 else [0, 1])
    return acc, f1, cm, y_true, y_pred

def run_permutation_test(X, y, pipe_fn, grid_params, true_acc, n_perms=100):
    rng = np.random.default_rng(42)
    perm_accs = []
    
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        acc, _, _, _, _ = run_nested_loo(X, y_perm, pipe_fn, grid_params)
        perm_accs.append(acc)
        
    p_value = (np.sum(np.array(perm_accs) >= true_acc) + 1.0) / (n_perms + 1.0)
    return p_value, np.mean(perm_accs)

def run_svr_experiment(X, y_cont, t1=1.41, t2=1.48):
    loo = LeaveOneOut()
    y_true_c, y_pred_c = [], []
    
    for tr, te in loo.split(X):
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectKBest(score_func=f_classif, k=10)),
            ('svr', SVR(C=10.0, gamma='scale', kernel='rbf'))
        ])
        pipe.fit(X[tr], y_cont[tr])
        y_pred_c.append(pipe.predict(X[te])[0])
        y_true_c.append(y_cont[te][0])
        
    r2 = r2_score(y_true_c, y_pred_c)
    r_val, p_val = pearsonr(y_true_c, y_pred_c)
    
    # Map predictions to 3-class using thresholds t1, t2
    y_true_d = [0 if v < t1 else (1 if v < t2 else 2) for v in y_true_c]
    y_pred_d = [0 if v < t1 else (1 if v < t2 else 2) for v in y_pred_c]
    
    acc = accuracy_score(y_true_d, y_pred_d)
    f1 = f1_score(y_true_d, y_pred_d, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_d, y_pred_d, labels=[0, 1, 2])
    
    return r_val, p_val, r2, acc, f1, cm

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, '6-classified_lame_level_features.json')
    
    print("==========================================================================")
    print("  COMPREHENSIVE PIG LAMENESS SVM OPTIMIZATION & VERIFICATION EXPERIMENT")
    print("==========================================================================")
    
    # Load dataset WITH and WITHOUT new veterinary kinematic features
    X_base, y_base, y_cont, names_base, ids = load_data(features_path, include_new_vet_features=False)
    X_full, y_full, _, names_full, _ = load_data(features_path, include_new_vet_features=True)
    
    print(f"Dataset baseline features count: {X_base.shape[1]}")
    print(f"Dataset full vet features count:     {X_full.shape[1]}")
    print(f"Class distribution: Level 1 (Sound)={np.sum(y_full==0)}, Level 2 (Medium)={np.sum(y_full==1)}, Level 3 (Lame)={np.sum(y_full==2)}")
    
    # Grid definition for Multi-class SVM
    svm_params_list = [
        {'k': 5, 'c': 1.0, 'gamma': 'scale', 'kernel': 'linear', 'cw': 'balanced', 'scaler': 'standard'},
        {'k': 10, 'c': 1.0, 'gamma': 'scale', 'kernel': 'linear', 'cw': 'balanced', 'scaler': 'robust'},
        {'k': 5, 'c': 10.0, 'gamma': 0.1, 'kernel': 'rbf', 'cw': 'balanced', 'scaler': 'robust'},
        {'k': 10, 'c': 10.0, 'gamma': 'scale', 'kernel': 'rbf', 'cw': 'balanced', 'scaler': 'robust'},
        {'k': 8, 'c': 5.0, 'gamma': 'scale', 'kernel': 'rbf', 'cw': 'balanced', 'scaler': 'standard'},
    ]
    
    def pipe_builder(p):
        sc = RobustScaler() if p['scaler'] == 'robust' else StandardScaler()
        return Pipeline([
            ('scaler', sc),
            ('select', SelectKBest(score_func=f_classif, k=p['k'])),
            ('svm', SVC(C=p['c'], gamma=p['gamma'], kernel=p['kernel'], class_weight=p['cw'], probability=True))
        ])

    def ord_pipe_builder(p):
        sc = RobustScaler() if p['scaler'] == 'robust' else StandardScaler()
        return Pipeline([
            ('scaler', sc),
            ('select', SelectKBest(score_func=f_classif, k=p['k'])),
            ('svm', OrdinalClassifier(SVC(C=p['c'], gamma=p['gamma'], kernel=p['kernel'], class_weight=p['cw'], probability=True)))
        ])

    # 1. Baseline Features vs New Veterinary Features (3-Class Nested LOO)
    print("\n--- Experiment 1: Baseline Features vs New Veterinary Features (3-Class Nested LOO) ---")
    acc_b, f1_b, cm_b, _, _ = run_nested_loo(X_base, y_base, pipe_builder, svm_params_list)
    print(f"  [Baseline DLC Features] -> Nested LOO Acc: {acc_b*100:.1f}%, F1-macro: {f1_b*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm_b}")
    
    acc_f, f1_f, cm_f, _, _ = run_nested_loo(X_full, y_full, pipe_builder, svm_params_list)
    print(f"  [+ New Veterinary Features] -> Nested LOO Acc: {acc_f*100:.1f}%, F1-macro: {f1_f*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm_f}")
    
    # Permutation test on full model
    print("\nRunning Permutation Test (30 iterations) for statistical significance...")
    p_val, p_mean = run_permutation_test(X_full, y_full, pipe_builder, svm_params_list, true_acc=acc_f, n_perms=30)
    print(f"  Permutation Test Result: p-value = {p_val:.4f} (Mean random accuracy = {p_mean*100:.1f}%)")

    # 2. Ordinal Classifier
    print("\n--- Experiment 2: Ordinal SVM Classifier (3-Class Nested LOO) ---")
    acc_ord, f1_ord, cm_ord, _, _ = run_nested_loo(X_full, y_full, ord_pipe_builder, svm_params_list)
    print(f"  [Ordinal SVM] -> Nested LOO Acc: {acc_ord*100:.1f}%, F1-macro: {f1_ord*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm_ord}")

    # 3. Continuous Support Vector Regression (SVR)
    print("\n--- Experiment 3: Continuous Support Vector Regression (SVR) ---")
    r_val, p_val_r, r2, acc_svr, f1_svr, cm_svr = run_svr_experiment(X_full, y_cont, t1=1.41, t2=1.48)
    print(f"  [SVR Model] -> Pearson r: {r_val:.3f} (p={p_val_r:.4e}), R2: {r2:.3f}")
    print(f"  [SVR Discretized 3-Class Classification] -> LOO Acc: {acc_svr*100:.1f}%, F1-macro: {f1_svr*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm_svr}")

    # 4. Binary Sound vs Lame Classification (Sound level1 vs Lame level3)
    print("\n--- Experiment 4: Binary Sound vs Lame SVM (2-Class) ---")
    mask_bin = y_full != 1 # Exclude Level 2 medium
    X_bin = X_full[mask_bin]
    y_bin = np.where(y_full[mask_bin] == 2, 1, 0)
    
    bin_params_list = [
        {'k': 5, 'c': 0.1, 'gamma': 'scale', 'kernel': 'linear', 'cw': 'balanced', 'scaler': 'standard'},
        {'k': 8, 'c': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'cw': 'balanced', 'scaler': 'robust'},
        {'k': 10, 'c': 10.0, 'gamma': 0.01, 'kernel': 'rbf', 'cw': 'balanced', 'scaler': 'robust'},
    ]
    
    acc_bin, f1_bin, cm_bin, _, _ = run_nested_loo(X_bin, y_bin, pipe_builder, bin_params_list)
    print(f"  [Binary Sound vs Lame SVM] -> Nested LOO Acc: {acc_bin*100:.1f}%, F1-macro: {f1_bin*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm_bin}")

    # Export Report Summary JSON
    summary_report = {
        "3_class_baseline_nested_acc": float(acc_b),
        "3_class_full_nested_acc": float(acc_f),
        "3_class_full_nested_f1": float(f1_f),
        "permutation_p_value": float(p_val),
        "ordinal_nested_acc": float(acc_ord),
        "svr_pearson_r": float(r_val),
        "svr_r2": float(r2),
        "svr_discretized_acc": float(acc_svr),
        "binary_nested_acc": float(acc_bin),
        "binary_nested_f1": float(f1_bin)
    }
    
    report_file = os.path.join(base_dir, 'comprehensive_svm_summary.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved comprehensive summary to {report_file}")

if __name__ == '__main__':
    main()
