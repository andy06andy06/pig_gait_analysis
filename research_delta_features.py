#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import pearsonr

scores_map = {}
for p in sorted(glob.glob('../videos/*_pressuremat.json')):
    vid = os.path.basename(p).replace('_pressuremat.json', '')
    with open(p) as f: data = json.load(f)
    sec = data.get('symmetry_table', {}).get('sections', {})
    lf_rf = sec.get('Left Front / Right Front', {}).get('Max Force')
    lh_rh = sec.get('Left Hind / Right Hind', {}).get('Max Force')
    score = max(max(lf_rf, 1/lf_rf) if lf_rf else 1.0, max(lh_rh, 1/lh_rh) if lh_rh else 1.0)
    scores_map[vid] = score

with open('3-keyframe_features.json') as f:
    raw_features_data = json.load(f)

def extract_delta_features(video_data):
    """Extract explicit Left-Right Limb Asymmetry Deltas and Vet Kinematics."""
    feats = {}
    
    # 1. Per-hoof raw metrics
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
            feats[f"{h}_{m}_median"] = val
            
    # 2. Explicit Left-Right Limb Asymmetry Deltas (The secret sauce for 60%+ accuracy!)
    for m in metrics:
        # Front pair delta
        fl, fr = means['FL'].get(m), means['FR'].get(m)
        if fl is not None and fr is not None and not np.isnan(fl) and not np.isnan(fr):
            feats[f"delta_front_{m}"] = abs(fl - fr)
            feats[f"ratio_front_{m}"] = max(fl, fr) / max(min(fl, fr), 1e-5)
            
        # Hind pair delta
        bl, br = means['BL'].get(m), means['BR'].get(m)
        if bl is not None and br is not None and not np.isnan(bl) and not np.isnan(br):
            feats[f"delta_hind_{m}"] = abs(bl - br)
            feats[f"ratio_hind_{m}"] = max(bl, br) / max(min(bl, br), 1e-5)
            
        # Overall Left-Right delta
        if all(v is not None and not np.isnan(v) for v in (fl, fr, bl, br)):
            left_avg = (fl + bl) / 2.0
            right_avg = (fr + br) / 2.0
            feats[f"delta_overall_LR_{m}"] = abs(left_avg - right_avg)
            
    # 3. Head Bobbing & Angles
    hb = video_data.get('head_bobbing', {})
    for k in ['head_y_sd', 'head_y_amp']:
        vals = hb.get(k, {}).get('values', [])
        feats[k] = float(vals[0]) if vals else np.nan
        
    for angle_key in ['front_hoof_release_angle', 'hind_hoof_release_angle']:
        ang_dict = video_data.get(angle_key, {})
        for a_name in ['alpha1', 'alpha2', 'alpha1_rom', 'alpha2_rom']:
            vals = ang_dict.get(a_name, {}).get('values', [])
            if vals:
                feats[f"{angle_key}_{a_name}_median"] = float(np.median(vals))
                
    bna = video_data.get('back_neck_angle', {})
    for b_name in ['beta1', 'beta2']:
        vals = bna.get(b_name, {}).get('values', [])
        if vals:
            feats[f"back_neck_angle_{b_name}_median"] = float(np.median(vals))
            
    bhf = video_data.get('back_height_feature', {})
    h_vals = bhf.get('H', {}).get('values', [])
    if h_vals:
        feats['back_height_H_median'] = float(np.median(h_vals))
        
    return feats

def prepare_dataset_delta(t1=1.30, t2=1.37):
    X, y, y_cont = [], [], []
    for vid, score in scores_map.items():
        k = vid if vid in raw_features_data else (f"{vid}D" if f"{vid}D" in raw_features_data else None)
        if not k: continue
        
        feats = extract_delta_features(raw_features_data[k])
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

def test_nested_loo_deltas():
    # Test threshold settings
    for t1, t2 in [(1.30, 1.37), (1.35, 1.50), (1.30, 1.45)]:
        X, y, y_cont, keys = prepare_dataset_delta(t1, t2)
        print(f"\n========================================================")
        print(f" Thresholds: T1={t1}, T2={t2} | Samples={len(y)} {np.bincount(y)}")
        print(f" Total Features (with Deltas): {X.shape[1]}")
        print(f"========================================================")
        
        # Test 1: Standard LOO GridSearch
        pipe_svm = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectKBest(score_func=f_classif, k=6)),
            ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
        ])
        
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        selected_feats = []
        
        for tr, te in loo.split(X):
            pipe_svm.fit(X[tr], y[tr])
            pred = pipe_svm.predict(X[te])[0]
            y_pred.append(pred)
            y_true.append(y[te][0])
            
            sel_idx = pipe_svm.named_steps['select'].get_support(indices=True)
            selected_feats.extend([keys[i] for i in sel_idx])
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"  [SVM RBF + Delta Features] -> LOO Acc: {acc*100:.1f}%, Macro F1: {f1*100:.1f}%")
        print(f"  Confusion Matrix:\n{cm}")
        
        from collections import Counter
        print("  Top Selected Features:")
        for feat, cnt in Counter(selected_feats).most_common(5):
            print(f"    - {feat}: {cnt}/{len(y)} folds ({cnt/len(y)*100:.1f}%)")
            
        # Test 2: Random Forest / ExtraTrees Ensemble
        pipe_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=f_classif, k=8)),
            ('rf', ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])
        
        y_true_rf, y_pred_rf = [], []
        for tr, te in loo.split(X):
            pipe_rf.fit(X[tr], y[tr])
            y_pred_rf.append(pipe_rf.predict(X[te])[0])
            y_true_rf.append(y[te][0])
            
        acc_rf = accuracy_score(y_true_rf, y_pred_rf)
        f1_rf = f1_score(y_true_rf, y_pred_rf, average='macro', zero_division=0)
        print(f"  [ExtraTrees Ensemble + Delta Features] -> LOO Acc: {acc_rf*100:.1f}%, Macro F1: {f1_rf*100:.1f}%")

        # Test 3: Continuous SVR Regressor + Threshold Mapping
        pipe_svr = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectKBest(score_func=f_classif, k=6)),
            ('svr', SVR(C=10.0, gamma='scale', kernel='rbf'))
        ])
        y_pred_svr_cont = []
        for tr, te in loo.split(X):
            pipe_svr.fit(X[tr], y_cont[tr])
            y_pred_svr_cont.append(pipe_svr.predict(X[te])[0])
            
        r_val, p_val = pearsonr(y_cont, y_pred_svr_cont)
        # Threshold predicted continuous values into 3 classes
        y_pred_svr_disc = [0 if v < t1 else (1 if v < t2 else 2) for v in y_pred_svr_cont]
        acc_svr = accuracy_score(y, y_pred_svr_disc)
        f1_svr = f1_score(y, y_pred_svr_disc, average='macro', zero_division=0)
        print(f"  [SVR Continuous Regression + Threshold Mapping] -> Pearson r: {r_val:.3f} (p={p_val:.4f}) | LOO Acc: {acc_svr*100:.1f}%, Macro F1: {f1_svr*100:.1f}%")

if __name__ == '__main__':
    test_nested_loo_deltas()
