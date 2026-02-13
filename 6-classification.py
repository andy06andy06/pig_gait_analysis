import os
import shutil
import glob
import json
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

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
        
        if isinstance(v, dict):
            features.update(get_features_from_dict(v, f"{prefix}{k}_"))
        elif isinstance(v, list):
            # check if list of numbers
            if v and isinstance(v[0], (int, float)):
                features[f"{prefix}{k}_mean"] = np.mean(v)
                features[f"{prefix}{k}_std"] = np.std(v)
            else:
                pass
        elif isinstance(v, (int, float)):
            features[f"{prefix}{k}"] = v
    return features

def prepare_dataset(classified_features):
    X = []
    y = []
    
    # Process Lame (label 1)
    for vid, data in classified_features.get('lame', {}).items():
        feats = get_features_from_dict(data)
        X.append(feats)
        y.append(1)
        
    # Process Sound (label 0)
    for vid, data in classified_features.get('sound', {}).items():
        feats = get_features_from_dict(data)
        X.append(feats)
        y.append(0)
        
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

def visualize_2d_decision_boundary(X, y, best_model, output_dir):
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
        svm_params = best_model.named_steps['svm'].get_params()
        
        # Construct a new SVM instance with the same parameters
        # We only keep parameters that are relevant to the SVC constructor
        valid_params = {}
        for k, v in svm_params.items():
            # Filter out pipeline-specific or irrelevant attributes if any (usually get_params returns valid init params)
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
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
        
        # Plot the training points
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=80)
        
        # Labeling
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        plt.title('SVM Decision Boundary in 2D PCA Space')
        
        # Legend
        handles, labels = scatter.legend_elements()
        plt.legend(handles, ['Sound', 'Lame'], loc="upper right", title="Classes")
        
        # Save
        save_path = os.path.join(output_dir, 'svm_decision_boundary_2d.png')
        plt.savefig(save_path)
        plt.close()
        print(f"2D Visualization saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating 2D visualization: {e}")
        import traceback
        traceback.print_exc()

def SVM_classification(feature_file_path):
    print(f"Loading features from {feature_file_path}...")
    try:
        with open(feature_file_path, 'r') as f:
            classified_features = json.load(f)
    except FileNotFoundError:
        print(f"Error: {feature_file_path} not found.")
        return 0.0

    X, y, feature_names = prepare_dataset(classified_features)
    
    if len(X) == 0:
        print("Error: No data found to classify.")
        return 0.0

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)} (0=Sound, 1=Lame)")
    
    # Define pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif)),
        ('svm', SVC(probability=True))
    ])

    # Define parameter grid
    param_grid = {
        'select__k': [5, 10, 20, 'all'],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf', 'linear', 'poly'],
        'svm__class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}]
    }

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
    
    target_names = ['Sound', 'Lame']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # --- Visualization ---
    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(feature_file_path), 'classification')
    os.makedirs(output_dir, exist_ok=True)

    # Save GridSearch results
    results_path = os.path.join(output_dir, 'gridsearch_results.json')
    
    gridsearch_results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "selected_features": selected_features,
        "final_accuracy": accuracy,
        "classification_report": classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    }

    def default_converter(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    try:
        with open(results_path, 'w') as f:
            json.dump(gridsearch_results, f, indent=4, default=default_converter)
        print(f"GridSearch results saved to {results_path}")
    except Exception as e:
        print(f"Error saving gridsearch results: {e}")
    
    # 1. Confusion Matrix
    try:
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=target_names, cmap=plt.cm.Blues)
        cm_path = os.path.join(output_dir, 'svm_confusion_matrix.png')
        plt.title(f"SVM Best Model Confusion Matrix\nAcc: {accuracy:.2f}")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion Matrix saved to {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

    # 2. 2D Decision Boundary
    visualize_2d_decision_boundary(X, y, best_model, output_dir)
    
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

def main():
    classified_file_operation()

if __name__ == "__main__":
    main()
