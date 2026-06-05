#!/usr/bin/env python3
import os
import json
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import clone

class OversamplingPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None or transform == 'passthrough':
                continue
            Xt = transform.fit_transform(Xt, y)
            
        y = np.array(y)
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        new_indices = list(range(len(y)))
        rng = np.random.default_rng(42)
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            cls_count = len(cls_indices)
            if cls_count < max_count:
                extra_indices = rng.choice(cls_indices, size=(max_count - cls_count), replace=True)
                new_indices.extend(extra_indices)
                
        new_indices = np.array(new_indices)
        Xt_resampled = Xt[new_indices]
        y_resampled = y[new_indices]
        
        self.steps[-1][1].fit(Xt_resampled, y_resampled)
        return self

def get_features_from_dict(d, prefix=''):
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']:
            continue
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_median"] = np.median(v)
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = v
    return features

def prepare_dataset(classified_features, class_names):
    X, y = [], []
    for label, class_name in enumerate(class_names):
        for vid, data in classified_features.get(class_name, {}).items():
            feats = get_features_from_dict(data)
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

def evaluate_selector_and_scorer(pipe, param_grid, X, y):
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv_inner, scoring='f1_macro', n_jobs=-1)
    gs.fit(X, y)
    
    best_params = gs.best_params_
    best_model = clone(pipe)
    best_model.set_params(**best_params)
    
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = clone(best_model)
        clf.fit(X_train, y_train)
        y_true.append(y_test[0])
        y_pred.append(clf.predict(X_test)[0])
        
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    recalls = []
    for i in range(3):
        denom = np.sum(cm[i])
        rec = cm[i, i] / denom if denom > 0 else 0.0
        recalls.append(rec)
        
    return acc, recalls, best_params, cm

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, '6-classified_lame_level_features.json')
    with open(features_path) as f:
        classified_features = json.load(f)
        
    LAME_LEVEL_CLASSES = ["level1_sound", "level2_medium", "level3_lame"]
    X, y, feature_names = prepare_dataset(classified_features, LAME_LEVEL_CLASSES)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Sound={np.sum(y==0)}, Medium={np.sum(y==1)}, Lame={np.sum(y==2)}")
    print("Evaluating different feature selection mechanisms...\n")
    
    # 4 selectors
    selectors = {
        "ANOVA F-value": (
            SelectKBest(score_func=f_classif),
            {'select__k': [5, 10, 20, 'all']}
        ),
        "Mutual Information": (
            SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42)),
            {'select__k': [5, 10, 20, 'all']}
        ),
        "L1-regularized (LinearSVC)": (
            SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=2000)),
            {'select__threshold': ['mean', 'median', '1.25*mean']}
        ),
        "Tree-based (Random Forest)": (
            SelectFromModel(estimator=RandomForestClassifier(random_state=42)),
            {'select__threshold': ['mean', 'median', '1.25*mean']}
        )
    }
    
    # Base SVM parameters grid
    svm_param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf', 'linear', 'poly'],
        'svm__class_weight': [None, 'balanced']
    }
    
    results = {}
    
    for name, (selector, select_params) in selectors.items():
        print(f"--- Running evaluation for {name} ---")
        
        # We will use OversamplingPipeline
        pipe = OversamplingPipeline([
            ('scaler', StandardScaler()),
            ('select', selector),
            ('svm', SVC())
        ])
        
        # Combine grid
        grid = dict(svm_param_grid)
        grid.update(select_params)
        
        acc, recalls, best_params, cm = evaluate_selector_and_scorer(pipe, grid, X, y)
        results[name] = {
            "acc": acc,
            "recalls": recalls,
            "best_params": best_params,
            "cm": cm
        }
        
        print(f"  LOO Accuracy: {acc:.3f}")
        print(f"  Recalls (S/M/L): {recalls[0]:.3f} / {recalls[1]:.3f} / {recalls[2]:.3f}")
        print(f"  Best Params: {best_params}")
        print(f"  Confusion Matrix:\n{cm}")
        print()
        
    print("=== Comparison Table ===")
    print(f"{'Mechanism':<27} | {'LOO Acc':<7} | {'Recalls (S/M/L)':<18} | {'Best Selector Param'}")
    print("-" * 75)
    for name, r in results.items():
        rec_str = f"{r['recalls'][0]:.3f} / {r['recalls'][1]:.3f} / {r['recalls'][2]:.3f}"
        sel_param = {k: v for k, v in r['best_params'].items() if 'select__' in k}
        print(f"{name:<27} | {r['acc']:<7.3f} | {rec_str:<18} | {sel_param}")

if __name__ == '__main__':
    main()
