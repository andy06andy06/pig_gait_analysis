#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.stats import pearsonr
from collections import Counter

base_dir = os.path.dirname(os.path.abspath(__file__))

scores_map = {}
for p in sorted(glob.glob(os.path.join(base_dir, '../videos/*_pressuremat.json'))):
    vid = os.path.basename(p).replace('_pressuremat.json', '')
    with open(p) as f: data = json.load(f)
    sec = data.get('symmetry_table', {}).get('sections', {})
    lf_rf = sec.get('Left Front / Right Front', {}).get('Max Force')
    lh_rh = sec.get('Left Hind / Right Hind', {}).get('Max Force')
    score = max(max(lf_rf, 1/lf_rf) if lf_rf else 1.0, max(lh_rh, 1/lh_rh) if lh_rh else 1.0)
    scores_map[vid] = score

with open(os.path.join(base_dir, '3-standardized_keyframe_features.json')) as f:
    features_data = json.load(f)


def get_features(d, prefix='', exclude_symmetry=True):
    feats = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']: continue
        if exclude_symmetry and 'symmetry_ratio' in f'{prefix}{k}': continue
        if isinstance(v, dict): feats.update(get_features(v, f'{prefix}{k}_', exclude_symmetry))
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)): feats[f'{prefix}{k}_median'] = float(np.median(v))
        elif isinstance(v, (int, float)): feats[f'{prefix}{k}'] = float(v)
    return feats

def run_experiment(mode='binary', t_sound=1.35, t_lame=1.45, exclude_symmetry=True):
    X, y, ids = [], [], []
    for vid, score in scores_map.items():
        k = vid if vid in features_data else (f'{vid}D' if f'{vid}D' in features_data else None)
        if not k: continue
        
        if mode == 'binary':
            if score < t_sound:
                y.append(0)
                X.append(get_features(features_data[k], exclude_symmetry=exclude_symmetry))
                ids.append(vid)
            elif score >= t_lame:
                y.append(1)
                X.append(get_features(features_data[k], exclude_symmetry=exclude_symmetry))
                ids.append(vid)
        elif mode == '3class':
            if score < t_sound: label = 0
            elif score < t_lame: label = 1
            else: label = 2
            y.append(label)
            X.append(get_features(features_data[k], exclude_symmetry=exclude_symmetry))
            ids.append(vid)

    all_keys = sorted(list(set().union(*(d.keys() for d in X))))
    X_mat = np.array([[d.get(k, np.nan) for k in all_keys] for d in X], dtype=float)
    y = np.array(y)
    
    # Outer LOO
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    selected_features = []
    
    candidate_pipes = []
    for scaler_name, sc in [('std', StandardScaler()), ('rob', RobustScaler())]:
        for k_sel in [3, 5, 8]:
            for C_val in [0.1, 1.0, 10.0]:
                for kernel in ['linear', 'rbf']:
                    candidate_pipes.append({
                        'scaler': sc,
                        'k': k_sel,
                        'C': C_val,
                        'kernel': kernel
                    })

    for tr, te in loo.split(X_mat):
        X_tr, y_tr = X_mat[tr], y[tr]
        X_te, y_te = X_mat[te], y[te]
        
        means = np.nanmean(X_tr, axis=0)
        X_tr_c = np.where(np.isnan(X_tr), means, X_tr)
        X_te_c = np.where(np.isnan(X_te), means, X_te)
        
        # Inner Stratified 3-Fold CV
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_cv_score = -1.0
        best_cfg = candidate_pipes[0]
        
        for cfg in candidate_pipes:
            cv_scores = []
            for tr_in, val_in in skf.split(X_tr_c, y_tr):
                try:
                    pipe = Pipeline([
                        ('scaler', cfg['scaler']),
                        ('select', SelectKBest(score_func=f_classif, k=min(cfg['k'], X_tr_c.shape[1]))),
                        ('svm', SVC(kernel=cfg['kernel'], C=cfg['C'], class_weight='balanced'))
                    ])
                    pipe.fit(X_tr_c[tr_in], y_tr[tr_in])
                    p_val = pipe.predict(X_tr_c[val_in])
                    cv_scores.append(f1_score(y_tr[val_in], p_val, average='macro', zero_division=0))
                except Exception:
                    cv_scores.append(0.0)
            m_score = np.mean(cv_scores)
            if m_score > best_cv_score:
                best_cv_score = m_score
                best_cfg = cfg
                
        # Fit outer model with best inner config
        final_pipe = Pipeline([
            ('scaler', best_cfg['scaler']),
            ('select', SelectKBest(score_func=f_classif, k=min(best_cfg['k'], X_tr_c.shape[1]))),
            ('svm', SVC(kernel=best_cfg['kernel'], C=best_cfg['C'], class_weight='balanced'))
        ])
        final_pipe.fit(X_tr_c, y_tr)
        pred = final_pipe.predict(X_te_c)[0]
        
        y_pred.append(pred)
        y_true.append(y_te[0])
        
        sel_idx = final_pipe.named_steps['select'].get_support(indices=True)
        selected_features.extend([all_keys[i] for i in sel_idx])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    top_feats = Counter(selected_features).most_common(5)
    return acc, f1, cm, len(y), np.bincount(y), top_feats

def main():
    print("=========================================================================")
    print("  STRICT NESTED LOO CROSS-VALIDATION OPTIMIZATION EXPERIMENTS")
    print("=========================================================================")
    
    # Test 1: 2-Class Binary (Buffer Zone Excluded: Sound < 1.35 vs Lame >= 1.48)
    acc1, f1_1, cm1, n1, counts1, top1 = run_experiment(mode='binary', t_sound=1.35, t_lame=1.48, exclude_symmetry=True)
    print(f"\n1. Binary (Sound < 1.35 vs Lame >= 1.48, Buffer Excluded | N={n1} {counts1}):")
    print(f"   Nested LOO Accuracy = {acc1*100:.1f}%, F1-macro = {f1_1*100:.1f}%")
    print(f"   Confusion Matrix:\n{cm1}")
    print("   Top Selected Features:")
    for feat, cnt in top1: print(f"     - {feat}: {cnt}/{n1} folds ({cnt/n1*100:.1f}%)")

    # Test 2: 2-Class Binary (Sound < 1.40 vs Lame >= 1.40, All pigs included)
    acc2, f1_2, cm2, n2, counts2, top2 = run_experiment(mode='binary', t_sound=1.40, t_lame=1.40, exclude_symmetry=True)
    print(f"\n2. Binary (Sound < 1.40 vs Lame >= 1.40, All Included | N={n2} {counts2}):")
    print(f"   Nested LOO Accuracy = {acc2*100:.1f}%, F1-macro = {f1_2*100:.1f}%")
    print(f"   Confusion Matrix:\n{cm2}")
    print("   Top Selected Features:")
    for feat, cnt in top2: print(f"     - {feat}: {cnt}/{n2} folds ({cnt/n2*100:.1f}%)")

    # Test 3: 3-Class Severity Classification (Sound < 1.35, Medium 1.35-1.50, Lame >= 1.50)
    acc3, f1_3, cm3, n3, counts3, top3 = run_experiment(mode='3class', t_sound=1.35, t_lame=1.50, exclude_symmetry=True)
    print(f"\n3. 3-Class (Sound < 1.35 / Medium 1.35-1.50 / Lame >= 1.50 | N={n3} {counts3}):")
    print(f"   Nested LOO Accuracy = {acc3*100:.1f}%, F1-macro = {f1_3*100:.1f}%")
    print(f"   Confusion Matrix:\n{cm3}")
    print("   Top Selected Features:")
    for feat, cnt in top3: print(f"     - {feat}: {cnt}/{n3} folds ({cnt/n3*100:.1f}%)")

if __name__ == '__main__':
    main()
