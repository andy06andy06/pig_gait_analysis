import os
import shutil
import glob
import json
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.base import clone, BaseEstimator, ClassifierMixin

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=None):
        self.clf = clf
        self.clfs = []
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clfs = []
        n_classes = len(self.classes_)
        for i in range(n_classes - 1):
            binary_y = (y > i).astype(int)
            clf = clone(self.clf)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        probs = []
        for clf in self.clfs:
            probs.append(clf.predict_proba(X)[:, 1])
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
        class_probs = class_probs / row_sums
        
        return class_probs

    def predict(self, X):
        class_probs = self.predict_proba(X)
        return np.argmax(class_probs, axis=1)

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

def custom_recall_penalized_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = []
    for i in range(3):
        denom = np.sum(cm[i])
        rec = cm[i, i] / denom if denom > 0 else 0.0
        recalls.append(rec)
    
    if any(r == 0 for r in recalls):
        return 0.01 * f1_score(y_true, y_pred, average='macro', zero_division=0)
        
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

LAME_LEVEL_CLASSES = ["level1_sound", "level2_medium", "level3_lame"]
EXCLUDE_SYMMETRY_RATIO_FEATURES = False
EXCLUDE_RAW_LEG_FEATURES = False
PERMUTATION_TEST_N_PERMUTATIONS = 100
RANDOM_STATE = 42
REPEATED_TRAIN_TEST_N_SPLITS = 100
REPEATED_TRAIN_TEST_TEST_SIZE = 0.30

def build_svm_pipeline():
    return OversamplingPipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif)),
        ('svm', OrdinalClassifier(SVC(probability=True)))
    ])

def build_param_grid(n_features):
    # Optimized feature engineering: restrict k to 5 or 10 features to prevent overfitting and help minority class
    k_values = [3, 5, 10]
    return {
        'select__k': k_values,
        'svm__clf__C': [0.1, 1, 10, 100],
        'svm__clf__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'svm__clf__kernel': ['rbf', 'linear'],
        'svm__clf__class_weight': [None, 'balanced']
    }


def parse_video_id_from_h5(file_name):
    """Extract the video/pressuremat id before the DLC suffix in a DLC h5 filename."""
    base_name = os.path.basename(file_name)
    if "DLC_" in base_name:
        return base_name.split("DLC_", 1)[0]
    return os.path.splitext(base_name)[0]

def find_existing_feature_file(base_dir):
    """Prefer standardized keyframe features, then fall back to available feature JSON files."""
    candidates = [
        os.path.join(base_dir, '3-standardized_keyframe_features.json'),
        os.path.join(base_dir, '3-gait_features.json'),
        os.path.join(base_dir, '3-keyframe_features.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No feature JSON found. Tried: " + ", ".join(candidates))

def load_lame_level_ids_from_dataset(classified_dir):
    """Load class -> ids mapping by scanning videos/classified_video_new level folders."""
    class_ids = {}
    for class_name in LAME_LEVEL_CLASSES:
        class_dir = os.path.join(classified_dir, class_name)
        h5_files = sorted(glob.glob(os.path.join(class_dir, "*.h5")))
        ids = [parse_video_id_from_h5(path) for path in h5_files]
        class_ids[class_name] = ids
        print(f"Loaded {len(ids)} ids from {class_dir}")
    return class_ids

def extract_features_multiclass(class_ids, input_path, output_path):
    """Extract feature records for arbitrary class folders into a classified JSON."""
    print(f"Reading features from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    classified_data = {class_name: {} for class_name in class_ids}
    missing = []

    for class_name, ids in class_ids.items():
        print(f"Extracting features for {class_name}...")
        for vid in ids:
            key = find_key_robust(vid, data)
            if key:
                classified_data[class_name][vid] = data[key]
                print(f"  Found features for {vid}")
            else:
                missing.append((class_name, vid))
                print(f"  Warning: Features for {vid} not found in input JSON.")

    if missing:
        print(f"Warning: {len(missing)} ids were missing feature records.")

    print(f"Saving classified features to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(classified_data, f, indent=4)
    print("Multi-class feature extraction complete.")
    return classified_data

def determine_consistent_shuffle(ids, source_dir):
    shuffle_counts = Counter()
    shuffle_pattern = re.compile(r'(shuffle\d+)')
    
    print("Determining consistent shuffle file...")
    for video_id in ids:
        # Search for any h5 file
        pattern = os.path.join(source_dir, f"{video_id}*.h5")
        files = glob.glob(pattern)
        for f in files:
            match = shuffle_pattern.search(os.path.basename(f))
            if match:
                shuffle_counts[match.group(1)] += 1
                
    if not shuffle_counts:
        print("Warning: No shuffle patterns found in h5 files.")
        return None
        
    most_common_shuffle = shuffle_counts.most_common(1)[0][0]
    print(f"  Selected shuffle version: {most_common_shuffle} (found {shuffle_counts[most_common_shuffle]} times)")
    return most_common_shuffle

def copy_files(ids, source_dir, destination, target_shuffle=None, extension=".h5"):
    print(f"Processing files for destination: {destination}")
    for video_id in ids:
        # Search for the files in the videos directory
        pattern = os.path.join(source_dir, f"{video_id}*{extension}")
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No files found for ID {video_id} with extension {extension}")
            continue
            
        for file_path in files:
            file_name = os.path.basename(file_path)
            
            # Enforce shuffle match if provided
            if target_shuffle and target_shuffle not in file_name:
                continue

            dest_path = os.path.join(destination, file_name)
            
            # Check if file is already in destination
            if os.path.exists(dest_path):
                # print(f"File {file_name} already exists in {destination}, skipping.")
                continue
            
            print(f"Copying {file_name} to {destination}")
            try:
                shutil.copy(file_path, dest_path)
            except Exception as e:
                print(f"Error copying {file_name}: {e}")

def find_key_robust(vid, data_keys):
    if vid in data_keys:
        return vid
    # Try adding 'D' suffix (observed case for C0014)
    if f"{vid}D" in data_keys:
        print(f"  Mapped missing key {vid} to {vid}D")
        return f"{vid}D"
    return None

def extract_features(lame_ids, sound_ids, input_path, output_path):
    print(f"Reading features from {input_path}...")
    data = {}
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            data = json.load(f)
    else:
        # Fallback to other possible filenames
        alt_path = input_path.replace('gait_features', 'keyframe_features')
        if os.path.exists(alt_path):
            print(f"File not found at {input_path}, trying {alt_path}...")
            with open(alt_path, 'r') as f:
                data = json.load(f)
        else:
            print(f"Error: Neither {input_path} nor {alt_path} found.")
            return

    classified_data = {
        "lame": {},
        "sound": {}
    }

    print("Extracting features for Lame category...")
    for vid in lame_ids:
        key = find_key_robust(vid, data)
        if key:
            classified_data["lame"][vid] = data[key]
            print(f"  Found features for {vid}")
        else:
            print(f"  Warning: Features for {vid} not found in input JSON.")

    print("Extracting features for Sound category...")
    for vid in sound_ids:
        key = find_key_robust(vid, data)
        if key:
            classified_data["sound"][vid] = data[key]
            print(f"  Found features for {vid}")
        else:
            print(f"  Warning: Features for {vid} not found in input JSON.")

    print(f"Saving classified features to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(classified_data, f, indent=4)
    print("Feature extraction complete.")

def get_features_from_dict(d, prefix=''):
    features = {}
    for k, v in d.items():
        if k in ['unit', 'frames', 'legs', 'leg']: # Skip metadata
            continue
        if EXCLUDE_SYMMETRY_RATIO_FEATURES and 'symmetry_ratio' in f"{prefix}{k}":
            # Temporarily exclude all symmetry-ratio-derived features to avoid
            # circularity with labels created from pressure-mat symmetry tables.
            continue
        if EXCLUDE_RAW_LEG_FEATURES and prefix == '' and k in ['FL', 'FR', 'BL', 'BR']:
            continue
        
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            # check if list of numbers
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_median"] = np.median(v)
            else:
                pass
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = v
    return features

def prepare_dataset(classified_features, class_names=None):
    X = []
    y = []

    if class_names is None:
        if all(class_name in classified_features for class_name in LAME_LEVEL_CLASSES):
            class_names = LAME_LEVEL_CLASSES
        else:
            # Backward-compatible binary default for older classified feature JSON files.
            class_names = ['sound', 'lame']

    # Process arbitrary classes in a stable label order.
    for label, class_name in enumerate(class_names):
        for vid, data in classified_features.get(class_name, {}).items():
            feats = get_features_from_dict(data)
            X.append(feats)
            y.append(label)
        
    # Vectorize
    # Collect all keys to ensure consistent order
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

def load_feature_matrix(feature_file_path, target_names):
    """Load classified features and convert them to X/y/feature_names."""
    with open(feature_file_path, 'r') as f:
        classified_features = json.load(f)
    return prepare_dataset(classified_features, target_names)

def default_converter(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def run_nested_leave_one_out_cv(X, y, feature_names, target_names, output_dir):
    """Run nested LOO CV: outer LOO for evaluation, inner LOO for model selection."""
    print("\nRunning Nested Leave-One-Out CV...")
    outer_cv = LeaveOneOut()
    y_true, y_pred = [], []
    fold_details = []
    param_counter = Counter()
    selected_feature_counter = Counter()

    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X), start=1):
        print(f"  Nested outer fold {fold_idx}/{len(y)}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_grid = GridSearchCV(
            build_svm_pipeline(),
            build_param_grid(X.shape[1]),
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0,
        )
        inner_grid.fit(X_train, y_train)
        pred = inner_grid.predict(X_test)[0]

        selected_indices = inner_grid.best_estimator_.named_steps['select'].get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        for feat in selected_features:
            selected_feature_counter[feat] += 1
        param_key = json.dumps(inner_grid.best_params_, sort_keys=True, default=default_converter)
        param_counter[param_key] += 1

        y_true.append(y_test[0])
        y_pred.append(pred)
        fold_details.append({
            "fold": fold_idx,
            "test_index": int(test_index[0]),
            "true_label": int(y_test[0]),
            "pred_label": int(pred),
            "correct": bool(pred == y_test[0]),
            "best_params": inner_grid.best_params_,
            "inner_best_score": float(inner_grid.best_score_),
            "selected_features": selected_features,
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    nested_results = {
        "method": "Nested Leave-One-Out CV",
        "outer_cv": "LeaveOneOut",
        "inner_cv": "LeaveOneOut",
        "accuracy": accuracy,
        "target_names": target_names,
        "class_counts": {class_name: int(np.sum(y == i)) for i, class_name in enumerate(target_names)},
        "classification_report": report,
        "best_param_frequency": {
            k: v for k, v in param_counter.most_common()
        },
        "selected_feature_frequency": {
            k: v for k, v in selected_feature_counter.most_common()
        },
        "fold_details": fold_details,
    }

    nested_path = os.path.join(output_dir, 'nested_loo_results.json')
    with open(nested_path, 'w') as f:
        json.dump(nested_results, f, indent=4, default=default_converter)
    print(f"Nested LOO accuracy: {accuracy:.3f}")
    print(f"Nested LOO results saved to {nested_path}")
    return nested_results

def run_label_permutation_test(X, y, best_params, target_names, output_dir, n_permutations=PERMUTATION_TEST_N_PERMUTATIONS):
    """Run a label permutation test using the selected SVM configuration."""
    print(f"\nRunning Label Permutation Test ({n_permutations} permutations)...")
    rng = np.random.default_rng(RANDOM_STATE)
    cv = LeaveOneOut()

    observed_model = build_svm_pipeline()
    observed_model.set_params(**best_params)
    observed_scores = cross_val_score(observed_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    observed_accuracy = float(np.mean(observed_scores))

    permutation_accuracies = []
    for i in range(n_permutations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")
        y_perm = rng.permutation(y)
        model = build_svm_pipeline()
        model.set_params(**best_params)
        scores = cross_val_score(model, X, y_perm, cv=cv, scoring='accuracy', n_jobs=-1)
        permutation_accuracies.append(float(np.mean(scores)))

    permutation_accuracies = np.array(permutation_accuracies)
    p_value = float((np.sum(permutation_accuracies >= observed_accuracy) + 1) / (n_permutations + 1))

    permutation_results = {
        "method": "Label Permutation Test",
        "n_permutations": n_permutations,
        "random_state": RANDOM_STATE,
        "cv": "LeaveOneOut",
        "best_params_used": best_params,
        "observed_accuracy": observed_accuracy,
        "permutation_accuracy_mean": float(np.mean(permutation_accuracies)),
        "permutation_accuracy_std": float(np.std(permutation_accuracies)),
        "permutation_accuracy_min": float(np.min(permutation_accuracies)),
        "permutation_accuracy_max": float(np.max(permutation_accuracies)),
        "p_value": p_value,
        "permutation_accuracies": permutation_accuracies.tolist(),
        "target_names": target_names,
    }

    permutation_path = os.path.join(output_dir, 'label_permutation_test_results.json')
    with open(permutation_path, 'w') as f:
        json.dump(permutation_results, f, indent=4, default=default_converter)
    print(f"Observed LOO accuracy: {observed_accuracy:.3f}")
    print(f"Permutation mean accuracy: {permutation_results['permutation_accuracy_mean']:.3f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Permutation results saved to {permutation_path}")
    return permutation_results

def run_repeated_stratified_train_test_evaluation(
    X,
    y,
    feature_names,
    target_names,
    output_dir,
    n_splits=REPEATED_TRAIN_TEST_N_SPLITS,
    test_size=REPEATED_TRAIN_TEST_TEST_SIZE,
):
    """Repeated stratified hold-out evaluation with grid search only on train data."""
    print(
        f"\nRunning Repeated Stratified Train/Test Evaluation "
        f"({n_splits} splits, test_size={test_size})..."
    )
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    split_details = []
    accuracies = []
    macro_f1_scores = []
    weighted_f1_scores = []
    aggregate_cm = np.zeros((len(target_names), len(target_names)), dtype=int)
    param_counter = Counter()
    selected_feature_counter = Counter()

    for split_idx, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        if split_idx == 1 or split_idx % 10 == 0:
            print(f"  Train/test split {split_idx}/{n_splits}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        min_class_count = int(np.min(np.bincount(y_train, minlength=len(target_names))))
        inner_splits = max(2, min(3, min_class_count))
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE + split_idx,
        )

        grid_search = GridSearchCV(
            build_svm_pipeline(),
            build_param_grid(X.shape[1]),
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(target_names))))
        aggregate_cm += cm

        selected_indices = grid_search.best_estimator_.named_steps['select'].get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        for feat in selected_features:
            selected_feature_counter[feat] += 1
        param_key = json.dumps(grid_search.best_params_, sort_keys=True, default=default_converter)
        param_counter[param_key] += 1

        accuracies.append(float(accuracy))
        macro_f1_scores.append(float(macro_f1))
        weighted_f1_scores.append(float(weighted_f1))
        split_details.append({
            "split": split_idx,
            "train_size": int(len(train_index)),
            "test_size": int(len(test_index)),
            "train_class_counts": {
                target_names[i]: int(np.sum(y_train == i)) for i in range(len(target_names))
            },
            "test_class_counts": {
                target_names[i]: int(np.sum(y_test == i)) for i in range(len(target_names))
            },
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "best_params": grid_search.best_params_,
            "inner_best_score": float(grid_search.best_score_),
            "selected_features": selected_features,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "test_indices": test_index.tolist(),
        })

    results = {
        "method": "Repeated Stratified Train/Test Evaluation",
        "n_splits": n_splits,
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "inner_cv": "StratifiedKFold(n_splits=min(3, min_train_class_count))",
        "target_names": target_names,
        "class_counts": {target_names[i]: int(np.sum(y == i)) for i in range(len(target_names))},
        "feature_count": int(X.shape[1]),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_min": float(np.min(accuracies)),
        "accuracy_max": float(np.max(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
        "weighted_f1_mean": float(np.mean(weighted_f1_scores)),
        "weighted_f1_std": float(np.std(weighted_f1_scores)),
        "aggregate_confusion_matrix": aggregate_cm.tolist(),
        "best_param_frequency": {k: v for k, v in param_counter.most_common()},
        "selected_feature_frequency": {k: v for k, v in selected_feature_counter.most_common()},
        "split_details": split_details,
    }

    results_path = os.path.join(output_dir, 'repeated_train_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=default_converter)
    print(f"Repeated train/test mean accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"Repeated train/test mean macro F1: {results['macro_f1_mean']:.3f} ± {results['macro_f1_std']:.3f}")
    print(f"Repeated train/test results saved to {results_path}")

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=aggregate_cm,
            display_labels=target_names,
        )
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title(
            "Repeated Stratified Train/Test Aggregate Confusion Matrix\n"
            f"Acc: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}",
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        plt.xticks(rotation=20, ha='right')
        cm_path = os.path.join(output_dir, 'repeated_train_test_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Repeated train/test confusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"Error generating repeated train/test confusion matrix: {e}")

    return results

def visualize_2d_decision_boundary(X, y, best_model, output_dir, target_names=None):
    print("\nGenerating 2D Decision Boundary visualization...")
    try:
        # 1. Transform data using the best pipeline's preprocessing steps (Scaler + Selection)
        # We need to use the already fitted scaler and selector from best_model
        
        # Access steps directly
        scaler = best_model.named_steps['scaler']
        selector = best_model.named_steps['select']
        
        X_scaled = scaler.transform(X)
        X_selected = selector.transform(X_scaled)
        
        # 2. PCA to 2D
        # Note: If fewer than 2 features selected, PCA might fail or be trivial.
        if X_selected.shape[1] < 2:
            print("Warning: Fewer than 2 features selected. Skipping 2D visualization.")
            return

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_selected)
        
        print(f"Explained Variance Ratio (First 2 components): {pca.explained_variance_ratio_}")
        
        # 3. Train a 2D SVM for visualization
        # We try to use the same params as the best model
        svm_step = best_model.named_steps['svm']
        if isinstance(svm_step, OrdinalClassifier):
            nested_svm_params = svm_step.clf.get_params()
            clf_2d = OrdinalClassifier(SVC(**nested_svm_params))
        else:
            svm_params = svm_step.get_params()
            valid_params = {}
            for k, v in svm_params.items():
                valid_params[k] = v
            clf_2d = SVC(**valid_params)
        clf_2d.fit(X_pca, y)
        
        # 4. Plotting
        plt.figure(figsize=(10, 8))
        
        # Create meshgrid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        
        # Resolution of mesh
        h = max((x_max - x_min) / 200, (y_max - y_min) / 200)
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict for each point in mesh
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Put the result into a color plot
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.22)
        
        # Plot the training points
        plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm,
            edgecolors='k', linewidths=1.1, s=170, alpha=0.95
        )
        
        # Labeling
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontsize=13)
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontsize=13)
        plt.title('SVM Decision Boundary in 2D PCA Space', fontsize=15)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Legend
        if target_names is None:
            target_names = [str(label) for label in sorted(np.unique(y))]
        cmap = plt.cm.get_cmap('coolwarm', len(target_names))
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=class_name,
                   markerfacecolor=cmap(i), markeredgecolor='k',
                   markeredgewidth=1.1, markersize=12)
            for i, class_name in enumerate(target_names)
        ]
        plt.legend(
            handles=legend_handles,
            loc="upper right",
            title="Classes",
            fontsize=12,
            title_fontsize=13,
            framealpha=0.95,
            borderpad=0.8,
            labelspacing=0.6
        )
        
        # Save
        save_path = os.path.join(output_dir, 'svm_decision_boundary_2d.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D Visualization saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating 2D visualization: {e}")
        import traceback
        traceback.print_exc()

def SVM_classification(feature_file_path, target_names=None, output_subdir='classification'):
    print(f"Loading features from {feature_file_path}...")
    try:
        with open(feature_file_path, 'r') as f:
            classified_features = json.load(f)
    except FileNotFoundError:
        print(f"Error: {feature_file_path} not found.")
        return 0.0

    if target_names is None:
        target_names = list(classified_features.keys())

    X, y, feature_names = prepare_dataset(classified_features, target_names)
    
    if len(X) == 0:
        print("Error: No data found to classify.")
        return 0.0

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {np.unique(y)}")
    print("Class mapping:")
    for label, class_name in enumerate(target_names):
        print(f"  {label}={class_name} ({int(np.sum(y == label))} samples)")
    
    # Define pipeline and parameter grid
    pipe = build_svm_pipeline()
    param_grid = build_param_grid(X.shape[1])

    # Cross-validation strategy
    cv_strategy = LeaveOneOut()
    
    print("Starting GridSearchCV with LeaveOneOut CV...")
    grid_search = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print("\nBest Parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV Score (F1 Macro): {grid_search.best_score_:.3f}")
    
    best_model = grid_search.best_estimator_
    
    # Analyze Feature Importance (if linear kernel or univariate selection)
    selected_indices = best_model.named_steps['select'].get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"\nSelected Features ({len(selected_features)}):")
    print(selected_features)

    # Predictions using LeaveOneOut on the best model to generate full report
    y_pred = []
    y_true = []
    
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Clone the best estimator to ensure we don't leak data, fit on training fold
        from sklearn.base import clone
        clf = clone(best_model)
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)[0]
        y_pred.append(pred)
        y_true.append(y_test[0])
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nFinal Model Performance (Best Params LOO):")
    print(f"Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names, zero_division=0))

    # --- Visualization ---
    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(feature_file_path), output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Save GridSearch results
    results_path = os.path.join(output_dir, 'gridsearch_results.json')
    
    gridsearch_results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "selected_features": selected_features,
        "final_accuracy": accuracy,
        "target_names": target_names,
        "class_counts": {class_name: int(np.sum(y == i)) for i, class_name in enumerate(target_names)},
        "classification_report": classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names, zero_division=0, output_dict=True)
    }

    try:
        with open(results_path, 'w') as f:
            json.dump(gridsearch_results, f, indent=4, default=default_converter)
        print(f"GridSearch results saved to {results_path}")
    except Exception as e:
        print(f"Error saving gridsearch results: {e}")
    
    # 1. Confusion Matrix
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            display_labels=target_names,
            cmap=plt.cm.Blues,
            ax=ax
        )

        cm_display.ax_.set_title(f"SVM Best Model Confusion Matrix\nAcc: {accuracy:.2f}", fontsize=16, fontweight='bold')
        cm_display.ax_.set_xlabel("Predicted label", fontsize=13)
        cm_display.ax_.set_ylabel("True label", fontsize=13)
        cm_display.ax_.tick_params(axis='both', labelsize=18)

        # Enlarge values inside each confusion-matrix cell
        if hasattr(cm_display, 'text_') and cm_display.text_ is not None:
            for txt in cm_display.text_.ravel():
                txt.set_fontsize(26)
                txt.set_fontweight('bold')

        cm_path = os.path.join(output_dir, 'svm_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion Matrix saved to {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

    # 2. 2D Decision Boundary
    visualize_2d_decision_boundary(X, y, best_model, output_dir, target_names)

    # 3. Strict validation checks
    strict_results = {
        "nested_leave_one_out": run_nested_leave_one_out_cv(
            X, y, feature_names, target_names, output_dir
        ),
        "label_permutation_test": run_label_permutation_test(
            X, y, grid_search.best_params_, target_names, output_dir
        ),
        "repeated_stratified_train_test": run_repeated_stratified_train_test_evaluation(
            X, y, feature_names, target_names, output_dir
        ),
    }
    strict_path = os.path.join(output_dir, 'strict_validation_summary.json')
    with open(strict_path, 'w') as f:
        json.dump(strict_results, f, indent=4, default=default_converter)
    print(f"Strict validation summary saved to {strict_path}")
    
    return accuracy

def classified_file_operation():
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, '../videos')
    classified_dir = os.path.join(videos_dir, 'classified_video')
    
    lame_dir = os.path.join(classified_dir, 'lame')
    sound_dir = os.path.join(classified_dir, 'sound')
    
    # Path to gait features
    gait_features_path = os.path.join(base_dir, '3-gait_features.json')
    output_features_path = os.path.join(base_dir, '6-classified_gait_features.json')

    # Ensure destination directories exist
    os.makedirs(lame_dir, exist_ok=True)
    os.makedirs(sound_dir, exist_ok=True)

    # Define the classification lists
    lame_ids = ['C0033-seg1-l', 'C0033-seg2-l', 'C0038-seg1', 'C0038-seg2', '1209045', 'C0004', 'C0008', 'C0014', 'C0015', 'C0019', 'C0020', 'C0040']
    
    sound_ids = ['C0014-seg1', 'C0014-seg2', 'C0015-seg1', 'C0015-seg2', 'C0016-seg1', 'C0016-seg2',
                 'C0017-seg1', 'C0017-seg2', 'C0019-seg1', 'C0019-seg2', 'C0021-seg1', 'C0021-seg2', 
                 'C0022-seg1', 'C0022-seg2', 'C0024-seg1', 'C0024-seg2', 'C0032-seg1', 'C0032-seg2', 
                 'C0033-seg1-s', 'C0033-seg2-s']
                #  '1118004', '1118006', '1118008', '1118011', '1118013', '1118017', '1118020', '1118021', 
                #  '1118023', '1118029', '1118030', '1118034', '1209038', '1209039', '1209043', '1209047', 
                #  '1209049', '1209052', '1209054', '1209055', '1209056', '1209059', '1209060', '1209061', 
                #  '1209062', '1209063', '1209065', '1209066', '1209068', '1209073', '1209075', '1209076', 
                #  '1209079', '1209080', '1209083', '1209085', '1209086', '1209087', 'B0018', 'B0026', 
                #  'B0037', 'B0038', 'B0039', 'B0045', 'B0058', 'B0059', 'B0064', 'B0067', 'C0001', 
                #  'C0002', 'C0006', 'C0009', 'C0013', 'C0017', 'C0021', 'C0022', 'C0035']

    # NEW: Determine consistent shuffle
    all_ids = lame_ids + sound_ids
    target_shuffle = determine_consistent_shuffle(all_ids, videos_dir)

    # 1. Copy files
    print("--- Step 1: Copying h5 files ---")
    copy_files(lame_ids, videos_dir, lame_dir, target_shuffle)
    copy_files(sound_ids, videos_dir, sound_dir, target_shuffle)
    
    # 2. Extract features
    print("\n--- Step 2: Extracting features ---")
    extract_features(lame_ids, sound_ids, gait_features_path, output_features_path)
    
    # 3. SVM Classification
    print("\n--- Step 3: SVM Classification ---")
    SVM_classification(output_features_path)

def lame_level_classification_operation():
    """Run 3-class SVM classification using videos/classified_video_new."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, '../videos')
    classified_dir = os.path.join(videos_dir, 'classified_video_new')

    print("--- Step 1: Loading classified_video_new IDs ---")
    class_ids = load_lame_level_ids_from_dataset(classified_dir)
    total_ids = sum(len(ids) for ids in class_ids.values())
    if total_ids == 0:
        raise RuntimeError(f"No h5 files found in {classified_dir}")

    print("\n--- Step 2: Extracting features for lame levels ---")
    feature_file_path = find_existing_feature_file(base_dir)
    output_features_path = os.path.join(base_dir, '6-classified_lame_level_features.json')
    extract_features_multiclass(class_ids, feature_file_path, output_features_path)

    print("\n--- Step 3: 3-class SVM Classification ---")
    if EXCLUDE_RAW_LEG_FEATURES and EXCLUDE_SYMMETRY_RATIO_FEATURES:
        output_dir_name = 'classification_lame_level_posture_only'
    elif EXCLUDE_RAW_LEG_FEATURES:
        output_dir_name = 'classification_lame_level_no_raw_leg'
    else:
        output_dir_name = 'classification_lame_level'
    SVM_classification(
        output_features_path,
        target_names=LAME_LEVEL_CLASSES,
        output_subdir=output_dir_name,
    )

def main():
    lame_level_classification_operation()

if __name__ == "__main__":
    main()
