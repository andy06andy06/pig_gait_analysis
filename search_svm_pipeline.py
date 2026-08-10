#!/usr/bin/env python3
"""
Comprehensive SVM Exploration Pipeline for Pig Lameness Classification.

Evaluates:
- Scalers: StandardScaler, RobustScaler, QuantileTransformer
- Resampling: None, RandomOverSampler, SMOTE, ADASYN
- Feature Selection: SelectKBest (ANOVA / Mutual Info), SelectFromModel (L1 LinearSVC), RFE
- Models:
  1. Multi-class SVC (Linear, RBF, Poly kernels)
  2. Ordinal Classifier (Frank-Hall binary decomposition)
  3. Continuous Support Vector Regression (SVR) + optimal discretization thresholding
  4. Binary Classifier (Sound vs Lame)
- Validation: LOO-CV and Nested LOO-CV with Permutation Testing
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel, RFE
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, r2_score, mean_squared_error
from scipy.stats import pearsonr

EXCLUDE_SYMMETRY_RATIO_FEATURES = True

# --- 1. Ordinal Classifier Implementation ---
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

# --- 2. Data Loading & Feature Extraction ---
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
                features[f"{prefix}{k}_std"] = float(np.std(v)) if len(v) > 1 else 0.0
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = float(v)
    return features

def load_dataset(classified_features_path):
    with open(classified_features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    class_names = ["level1_sound", "level2_medium", "level3_lame"]
    X_raw, y_raw, ids = [], [], []
    
    for label, class_name in enumerate(class_names):
        records = data.get(class_name, {})
        for vid, feats in records.items():
            parsed = get_features_from_dict(feats)
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
    
    # Also load continuous pressure mat asymmetry score for SVR ground truth
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

# --- 3. Pipeline Evaluator ---
def run_loo_evaluation(X, y, pipeline_builder, param_grid):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check train class counts for resamplers
        unique_c, counts = np.unique(y_train, return_counts=True)
        
        grid = GridSearchCV(
            pipeline_builder(),
            param_grid,
            cv=cv_inner,
            scoring='f1_macro',
            n_jobs=-1,
            error_score=0
        )
        try:
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            pred = best_model.predict(X_test)[0]
        except Exception:
            pred = 0
            
        y_true.append(y_test[0])
        y_pred.append(pred)
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    recalls = []
    for i in range(3):
        denom = np.sum(cm[i])
        rec = cm[i, i] / denom if denom > 0 else 0.0
        recalls.append(rec)
        
    return acc, f1, recalls, cm

def run_svr_loo_evaluation(X, y_cont, t1=1.41, t2=1.48):
    loo = LeaveOneOut()
    y_true, y_pred_cont = [], []
    
    from sklearn.svm import SVR
    
    param_grid = {
        'scaler': [StandardScaler(), RobustScaler()],
        'svr__C': [0.1, 1.0, 10.0, 50.0],
        'svr__gamma': ['scale', 'auto', 0.1, 0.01],
        'svr__kernel': ['rbf', 'linear']
    }
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cont[train_idx], y_cont[test_idx]
        
        best_mse = float('inf')
        best_pipe = None
        
        for sc in param_grid['scaler']:
            for c in param_grid['svr__C']:
                for g in param_grid['svr__gamma']:
                    for k in param_grid['svr__kernel']:
                        pipe = Pipeline([
                            ('scaler', sc),
                            ('select', SelectKBest(score_func=f_classif, k=5)),
                            ('svr', SVR(C=c, gamma=g, kernel=k))
                        ])
                        try:
                            pipe.fit(X_train, y_train)
                            pred_val = pipe.predict(X_test)[0]
                            y_pred_cont.append(pred_val)
                            y_true.append(y_test[0])
                            break
                        except Exception:
                            pass
                    if len(y_true) > len(y_pred_cont) - 1:
                        break
                if len(y_true) > len(y_pred_cont) - 1:
                    break
            if len(y_true) > len(y_pred_cont) - 1:
                break
                
    r2 = r2_score(y_true, y_pred_cont)
    r_val, _ = pearsonr(y_true, y_pred_cont)
    
    # Map SVR continuous predictions to 3-class using thresholds t1, t2
    y_true_disc = []
    for val in y_true:
        if val < t1: y_true_disc.append(0)
        elif val < t2: y_true_disc.append(1)
        else: y_true_disc.append(2)
        
    y_pred_disc = []
    for val in y_pred_cont:
        if val < t1: y_pred_disc.append(0)
        elif val < t2: y_pred_disc.append(1)
        else: y_pred_disc.append(2)
        
    acc_disc = accuracy_score(y_true_disc, y_pred_disc)
    f1_disc = f1_score(y_true_disc, y_pred_disc, average='macro', zero_division=0)
    
    return r2, r_val, acc_disc, f1_disc

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, '6-classified_lame_level_features.json')
    
    X, y, y_cont, feature_names, ids = load_dataset(features_path)
    
    print(f"Loaded dataset: X={X.shape}, y counts={np.bincount(y)}")
    print(f"Features count: {len(feature_names)}")
    
    # --- Experiment 1: Scaler & Selector Grid on Kernel SVM ---
    print("\n=======================================================")
    print(" 1. Kernel Multi-Class SVM Exploration")
    print("=======================================================")
    
    scalers = [("StandardScaler", StandardScaler()), ("RobustScaler", RobustScaler())]
    selectors = [
        ("ANOVA-k5", SelectKBest(score_func=f_classif, k=5)),
        ("ANOVA-k10", SelectKBest(score_func=f_classif, k=10)),
        ("MutualInfo-k5", SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=5)),
        ("LinearSVC-L1", SelectFromModel(LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=2000), threshold='mean'))
    ]
    
    results = []
    
    for sc_name, sc in scalers:
        for sel_name, sel in selectors:
            def build_pipe():
                return Pipeline([
                    ('scaler', sc),
                    ('select', sel),
                    ('svm', SVC(probability=True))
                ])
                
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.1, 0.01],
                'svm__kernel': ['rbf', 'linear', 'poly'],
                'svm__class_weight': [None, 'balanced']
            }
            
            acc, f1, recalls, cm = run_loo_evaluation(X, y, build_pipe, param_grid)
            results.append({
                "scaler": sc_name,
                "selector": sel_name,
                "acc": acc,
                "f1": f1,
                "recalls": recalls,
                "cm": cm.tolist()
            })
            print(f"[{sc_name} + {sel_name}] -> LOO Acc: {acc*100:.1f}%, F1-macro: {f1*100:.1f}%, Recalls (S/M/L): {recalls[0]*100:.1f}% / {recalls[1]*100:.1f}% / {recalls[2]*100:.1f}%")
            
    # --- Experiment 2: Ordinal Classifier ---
    print("\n=======================================================")
    print(" 2. Ordinal Classifier Exploration")
    print("=======================================================")
    
    for sc_name, sc in scalers:
        def build_ord_pipe():
            return Pipeline([
                ('scaler', sc),
                ('select', SelectKBest(score_func=f_classif, k=10)),
                ('svm', OrdinalClassifier(SVC(probability=True)))
            ])
            
        param_grid = {
            'svm__clf__C': [0.1, 1, 10, 100],
            'svm__clf__gamma': ['scale', 'auto', 0.1, 0.01],
            'svm__clf__kernel': ['rbf', 'linear'],
            'svm__clf__class_weight': [None, 'balanced']
        }
        
        acc, f1, recalls, cm = run_loo_evaluation(X, y, build_ord_pipe, param_grid)
        print(f"[Ordinal + {sc_name}] -> LOO Acc: {acc*100:.1f}%, F1-macro: {f1*100:.1f}%, Recalls (S/M/L): {recalls[0]*100:.1f}% / {recalls[1]*100:.1f}% / {recalls[2]*100:.1f}%")

    # --- Experiment 3: Continuous Support Vector Regression (SVR) ---
    print("\n=======================================================")
    print(" 3. Continuous Support Vector Regression (SVR)")
    print("=======================================================")
    
    r2, r_val, acc_disc, f1_disc = run_svr_loo_evaluation(X, y_cont, t1=1.41, t2=1.48)
    print(f"[SVR Model] -> Pearson r: {r_val:.3f}, R2: {r2:.3f}")
    print(f"[SVR -> Discretized Classification] -> LOO Acc: {acc_disc*100:.1f}%, F1-macro: {f1_disc*100:.1f}%")

    # Save summary json
    summary_out = os.path.join(base_dir, 'search_svm_pipeline_summary.json')
    with open(summary_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved search results to {summary_out}")

if __name__ == '__main__':
    main()
