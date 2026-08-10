#!/usr/bin/env python3
import os
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC, SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.stats import pearsonr

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

def prepare_data(classified_features_path):
    with open(classified_features_path) as f:
        data = json.load(f)
    class_names = ["level1_sound", "level2_medium", "level3_lame"]
    X, y, ids = [], [], []
    for label, c in enumerate(class_names):
        for vid, feats in data.get(c, {}).items():
            X.append(get_features_from_dict(feats))
            y.append(label)
            ids.append(vid)
    all_keys = sorted(list(set().union(*(d.keys() for d in X))))
    X_mat = np.array([[d.get(k, np.nan) for k in all_keys] for d in X], dtype=float)
    means = np.nanmean(X_mat, axis=0)
    X_clean = np.where(np.isnan(X_mat), means, X_mat)
    return X_clean, np.array(y), all_keys, ids

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, '6-classified_lame_level_features.json')
    X, y, feature_names, ids = prepare_data(features_path)
    
    print("=" * 65)
    print(f"  PIG LAMENESS CLASSIFICATION MULTI-MODEL VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"Total Samples: {len(y)} | Feature Count: {X.shape[1]}")
    print(f"Class Counts: Sound(L1)={np.sum(y==0)}, Medium(L2)={np.sum(y==1)}, Lame(L3)={np.sum(y==2)}")
    
    # 1. Optimal 3-Class SVM (Tuned parameters)
    pipe_3class = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(score_func=f_classif, k=8)),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
    ])
    
    loo = LeaveOneOut()
    y_true_3, y_pred_3 = [], []
    
    for tr, te in loo.split(X):
        pipe_3class.fit(X[tr], y[tr])
        y_pred_3.append(pipe_3class.predict(X[te])[0])
        y_true_3.append(y[te][0])
        
    acc_3 = accuracy_score(y_true_3, y_pred_3)
    f1_3 = f1_score(y_true_3, y_pred_3, average='macro')
    cm_3 = confusion_matrix(y_true_3, y_pred_3, labels=[0, 1, 2])
    
    print("\n--- 1. 3-Class SVM (Sound vs Medium vs Lame) ---")
    print(f"LOO Accuracy: {acc_3*100:.1f}%")
    print(f"Macro F1-Score: {f1_3*100:.1f}%")
    print(f"Confusion Matrix:\n{cm_3}")
    print("\nClassification Report:")
    print(classification_report(y_true_3, y_pred_3, target_names=["level1_sound", "level2_medium", "level3_lame"]))
    
    # 2. Optimal 2-Class Binary SVM (Sound vs Lame)
    mask_2 = y != 1 # Exclude transition buffer Level 2
    X_2 = X[mask_2]
    y_2 = np.where(y[mask_2] == 2, 1, 0)
    
    pipe_2class = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=5)),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced'))
    ])
    
    y_true_2, y_pred_2 = [], []
    for tr, te in loo.split(X_2):
        pipe_2class.fit(X_2[tr], y_2[tr])
        y_pred_2.append(pipe_2class.predict(X_2[te])[0])
        y_true_2.append(y_2[te][0])
        
    acc_2 = accuracy_score(y_true_2, y_pred_2)
    f1_2 = f1_score(y_true_2, y_pred_2, average='macro')
    cm_2 = confusion_matrix(y_true_2, y_pred_2, labels=[0, 1])
    
    print("--- 2. 2-Class Binary SVM (Sound vs Lame) ---")
    print(f"LOO Accuracy: {acc_2*100:.1f}%")
    print(f"Macro F1-Score: {f1_2*100:.1f}%")
    print(f"Confusion Matrix:\n{cm_2}")
    print("\nClassification Report:")
    print(classification_report(y_true_2, y_pred_2, target_names=["Sound", "Lame"]))

    # Save summary report artifact
    report_dict = {
        "3_class_accuracy": float(acc_3),
        "3_class_f1_macro": float(f1_3),
        "3_class_confusion_matrix": cm_3.tolist(),
        "2_class_accuracy": float(acc_2),
        "2_class_f1_macro": float(f1_2),
        "2_class_confusion_matrix": cm_2.tolist(),
    }
    with open(os.path.join(base_dir, 'final_verification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved final report to {os.path.join(base_dir, 'final_verification_report.json')}")

if __name__ == '__main__':
    main()
