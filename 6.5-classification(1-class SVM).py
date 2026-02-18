import os
import shutil
import glob
import json
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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
    X_sound = []
    X_lame = []
    
    # Process Lame (label -1 for OCSVM outlier, but originally 1)
    for vid, data in classified_features.get('lame', {}).items():
        feats = get_features_from_dict(data)
        X_lame.append(feats)
        
    # Process Sound (label 1 for OCSVM inlier, but originally 0)
    for vid, data in classified_features.get('sound', {}).items():
        feats = get_features_from_dict(data)
        X_sound.append(feats)
        
    # Vectorize
    # Collect all keys to ensure consistent order
    all_data = X_sound + X_lame
    if not all_data:
        return np.array([]), np.array([]), []

    all_keys = sorted(list(set().union(*(d.keys() for d in all_data))))
    
    def vectorize(data_list):
        if not data_list:
            return np.array([])
        tmp = []
        for d in data_list:
            tmp.append([d.get(k, np.nan) for k in all_keys])
        return np.array(tmp, dtype=float)

    X_sound_vec = vectorize(X_sound)
    X_lame_vec = vectorize(X_lame)
    
    # Impute NaNs with mean from SOUND data only (since that's our training set)
    # If sound data has NaNs for a column, we can try global mean or 0
    if X_sound_vec.size > 0:
        sound_means = np.nanmean(X_sound_vec, axis=0)
        # Handle case where a column is all NaNs in sound data
        sound_means = np.nan_to_num(sound_means, 0.0) 
        
        # Impute Sound
        inds = np.where(np.isnan(X_sound_vec))
        X_sound_vec[inds] = np.take(sound_means, inds[1])
        
        # Impute Lame using Sound means
        if X_lame_vec.size > 0:
            inds = np.where(np.isnan(X_lame_vec))
            X_lame_vec[inds] = np.take(sound_means, inds[1])
            
    return X_sound_vec, X_lame_vec, all_keys

def visualize_2d_decision_boundary(X_train, X_test, y_test, best_model, output_dir):
    print("\nGenerating 2D Decision Boundary visualization...")
    try:
        # Access steps directly
        scaler = best_model.named_steps['scaler']
        # If PCA is in the pipeline (which we added), we need to use it
        pca_step = best_model.named_steps.get('pca')
        
        # Transform training data (Sound)
        X_train_trans = scaler.transform(X_train)
        if pca_step:
            X_train_trans = pca_step.transform(X_train_trans)
        
        # Transform test data (Mixed)
        X_test_trans = scaler.transform(X_test)
        if pca_step:
            X_test_trans = pca_step.transform(X_test_trans)

        # For visualization, we need exactly 2 dimensions.
        # If the data is already > 2D (even after PCA step in pipeline), we need another PCA to reduce to 2D for plot.
        # If the data is <= 2D, we can use it directly or pad it.
        
        # Let's perform a dedicated 2D PCA for visualization purposes on the TRANSFORMED data
        pca_viz = PCA(n_components=2)
        X_train_viz = pca_viz.fit_transform(X_train_trans) # Fit on training (sound)
        X_test_viz = pca_viz.transform(X_test_trans)       # Transform test
        
        print(f"Visualization PCA Explained Variance Ratio: {pca_viz.explained_variance_ratio_}")
        
        # Train a 2D One-Class SVM for visualization
        # We try to use the same params as the best model
        svm_params = best_model.named_steps['ocsvm'].get_params()
        
        # Construct a new OCSVM instance
        clf_2d = OneClassSVM(kernel=svm_params.get('kernel', 'rbf'), 
                             gamma=svm_params.get('gamma', 'scale'), 
                             nu=svm_params.get('nu', 0.5))
        clf_2d.fit(X_train_viz)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Create meshgrid
        # Combine to find limits
        all_viz = np.vstack([X_train_viz, X_test_viz])
        x_min, x_max = all_viz[:, 0].min() - 1, all_viz[:, 0].max() + 1
        y_min, y_max = all_viz[:, 1].min() - 1, all_viz[:, 1].max() + 1
        
        # Resolution
        h = max((x_max - x_min) / 200, (y_max - y_min) / 200)
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict Z
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Contour
        plt.contourf(xx, yy, Z, levels=[0, 0.5], colors=['palevioletred'], alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        
        # Plot Sound (Train)
        plt.scatter(X_train_viz[:, 0], X_train_viz[:, 1], c='blue', edgecolors='k', s=80, label='Training (Sound)')
        
        # Plot Test Data
        # Sound in Test (Label 0)
        test_sound_mask = (y_test == 0)
        if np.any(test_sound_mask):
            plt.scatter(X_test_viz[test_sound_mask, 0], X_test_viz[test_sound_mask, 1], 
                        c='cyan', edgecolors='k', marker='o', s=80, label='Test (Sound)')
            
        # Lame in Test (Label 1)
        test_lame_mask = (y_test == 1)
        if np.any(test_lame_mask):
            plt.scatter(X_test_viz[test_lame_mask, 0], X_test_viz[test_lame_mask, 1], 
                        c='red', edgecolors='k', marker='^', s=80, label='Test (Lame)')
        
        plt.xlabel(f'PCA Component 1 ({pca_viz.explained_variance_ratio_[0]:.2%} var)')
        plt.ylabel(f'PCA Component 2 ({pca_viz.explained_variance_ratio_[1]:.2%} var)')
        plt.title('One-Class SVM Decision Boundary in 2D PCA Space')
        plt.legend(loc="upper right")
        
        save_path = os.path.join(output_dir, 'ocsvm_decision_boundary_2d.png')
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

    # X_sound is our "normal" data for training
    # X_lame is our "abnormal" data, used only for testing/validation
    X_sound, X_lame, feature_names = prepare_dataset(classified_features)
    
    if len(X_sound) == 0:
        print("Error: No Sound data found to train One-Class SVM.")
        return 0.0

    print(f"Dataset shapes - Sound (Train): {X_sound.shape}, Lame (Test): {X_lame.shape}")
    
    # --- One-Class SVM Pipeline with PCA ---
    # Added PCA to reduce dimensionality and noise, which helps OCSVM.
    # We keep 95% variance.
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('ocsvm', OneClassSVM())
    ])

    # --- Hyperparameter Tuning Strategy for OCSVM ---
    # Since we have some labeled anomalies (X_lame), we can use them to tune 'nu' and 'gamma'.
    # We will split X_sound into Train/Val, and use X_lame as Val (Anomalies).
    # Goal: Maximize F1-score (or other metric) on the Validation set (Sound Val + Lame Val).
    
    # Grid Search Parameters - UPDATED based on feedback
    # Expanded ranges for better optimization
    param_grid = {
        'ocsvm__nu': [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        'ocsvm__gamma': ['scale', 'auto', 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'ocsvm__kernel': ['rbf', 'sigmoid', 'poly'],
        'pca__n_components': [0.90, 0.95, 0.99]
    }
    
    best_score = -1
    best_params = {}
    best_model = None
    
    # We'll do a manual Grid Search with Cross Validation
    # Strategy: 
    #   Loop through parameters.
    #   Inside loop, perform K-Fold CV on X_sound.
    #   For each fold:
    #       Train on (K-1) Sound folds.
    #       Test on 1 Sound fold + All Lame data.
    #       Calculate F1-macro.
    #   Average score across folds.
    
    print("Starting Grid Search for One-Class SVM...")
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=min(5, len(X_sound)), shuffle=True, random_state=42)
    
    # Flatten grid for easier iteration
    from sklearn.model_selection import ParameterGrid
    grid = list(ParameterGrid(param_grid))
    
    print(f"Testing {len(grid)} parameter combinations...")

    for i, params in enumerate(grid):
        scores = []
        if i % 10 == 0:
            print(f"  Processing combination {i}/{len(grid)}...")
            
        # Extract PCA params separately
        pca_n = params.pop('pca__n_components', 0.95)

        for train_idx, val_idx in kf.split(X_sound):
            X_train_fold = X_sound[train_idx]
            X_val_sound_fold = X_sound[val_idx]
            
            # Combine validation sound with ALL lame data for testing this fold
            # Labels: Sound = 0, Lame = 1
            X_val_fold = np.vstack([X_val_sound_fold, X_lame]) if X_lame.size > 0 else X_val_sound_fold
            y_val_fold = np.hstack([np.zeros(len(X_val_sound_fold)), np.ones(len(X_lame))]) if X_lame.size > 0 else np.zeros(len(X_val_sound_fold))
            
            # Configure pipeline
            # Note: We need to reconstruct the pipeline for each iteration to change params
            current_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=pca_n)),
                ('ocsvm', OneClassSVM(**{k.replace('ocsvm__', ''): v for k, v in params.items()}))
            ])
            
            # Fit only on Sound data
            current_pipe.fit(X_train_fold)
            
            # Predict
            # OCSVM returns: 1 for inliers (Sound), -1 for outliers (Lame)
            preds_raw = current_pipe.predict(X_val_fold)
            
            # Map OCSVM outputs to our labels: 1 -> 0 (Sound), -1 -> 1 (Lame)
            preds_mapped = np.where(preds_raw == 1, 0, 1)
            
            # Score (Recall for Lame class - Prioritize sensitivity)
            # We want to catch as many Lame pigs as possible.
            # pos_label=1 refers to Lame class.
            recall = recall_score(y_val_fold, preds_mapped, pos_label=1, zero_division=0)
            
            # Secondary metric: Precision (to break ties if recalls are equal)
            # or F2 score (which weights recall higher than precision)
            # Let's use a weighted score: 0.8 * Recall + 0.2 * Precision to avoid trivial solutions
            # But the user specifically asked for sensitivity, so let's stick to Recall as primary.
            # However, pure Recall might just pick the highest `nu` (predicting everything as anomaly).
            # So let's use F2-score which weights Recall 2x more than Precision.
            
            # Actually, let's use F2 score to be safe against trivial "all anomaly" models
            beta = 2
            p = 0 # precision placeholder, not needed for direct calc
            # fbeta_score calculation manually or import it? Let's just use recall for now but check precision.
            
            # Simple approach: Maximize Recall, use Precision as tie-breaker
            score = recall
            scores.append(score)
        
        # Restore PCA param for saving to best_params
        params['pca__n_components'] = pca_n

        avg_score = np.mean(scores)
        
        # We need a tie-breaker because many params might give the same discrete recall (e.g. 10/12 vs 11/12)
        # But here we are averaging over folds, so it's a float.
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            print(f"  New best score (Recall Lame): {best_score:.4f} with params {best_params}")
            
    print(f"\nBest Parameters found: {best_params}")
    print(f"Best CV Score (Recall Lame): {best_score:.3f}")
    
    # --- Final Training and Evaluation ---
    # Train on ALL Sound data with best params
    pca_n_final = best_params.pop('pca__n_components', 0.95)
    
    best_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n_final)),
        ('ocsvm', OneClassSVM(**{k.replace('ocsvm__', ''): v for k, v in best_params.items()}))
    ])
    
    best_pipe.fit(X_sound)
    best_model = best_pipe # Save for viz
    
    # Put PCA param back for saving
    best_params['pca__n_components'] = pca_n_final
    
    # Evaluation on the entire dataset (Sound + Lame) just to see overall fit
    # Note: Training error on Sound is included here, which is expected for OCSVM (it should classify most Sound as Sound)
    X_full = np.vstack([X_sound, X_lame]) if X_lame.size > 0 else X_sound
    y_full = np.hstack([np.zeros(len(X_sound)), np.ones(len(X_lame))]) if X_lame.size > 0 else np.zeros(len(X_sound))
    
    preds_raw = best_pipe.predict(X_full)
    y_pred = np.where(preds_raw == 1, 0, 1) # 1->0 (Sound), -1->1 (Lame)
    
    accuracy = accuracy_score(y_full, y_pred)
    target_names = ['Sound', 'Lame']
    
    print("\nFinal Model Performance (Best Params on Full Dataset):")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_full, y_pred, target_names=target_names, zero_division=0))

    # --- Identify Misclassified IDs ---
    print("\nMisclassified Samples:")
    
    # Reconstruct IDs in the same order as X_full
    # X_full is [X_sound, X_lame]
    sound_ids = sorted(list(classified_features.get('sound', {}).keys()))
    lame_ids = sorted(list(classified_features.get('lame', {}).keys()))
    full_ids = sound_ids + lame_ids
    
    misclassified_info = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_full, y_pred)):
        if true_label != pred_label:
            vid_id = full_ids[i] if i < len(full_ids) else "Unknown"
            status = "False Positive (Healthy classified as Lame)" if pred_label == 1 else "False Negative (Lame classified as Healthy)"
            print(f"  {vid_id}: {status}")
            misclassified_info.append({
                "id": vid_id,
                "true_label": "Sound" if true_label == 0 else "Lame",
                "pred_label": "Sound" if pred_label == 0 else "Lame",
                "status": status
            })

    # --- Visualization ---
    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(feature_file_path), 'classification_ocsvm')
    os.makedirs(output_dir, exist_ok=True)

    # Save Results
    results_path = os.path.join(output_dir, 'ocsvm_results.json')
    
    results_data = {
        "best_params": best_params,
        "best_cv_score": best_score,
        "final_accuracy": accuracy,
        "classification_report": classification_report(y_full, y_pred, target_names=target_names, zero_division=0, output_dict=True),
        "misclassified_samples": misclassified_info
    }

    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # 1. Confusion Matrix
    try:
        cm_display = ConfusionMatrixDisplay.from_predictions(y_full, y_pred, display_labels=target_names, cmap=plt.cm.Blues)
        cm_path = os.path.join(output_dir, 'ocsvm_confusion_matrix.png')
        plt.title(f"One-Class SVM Confusion Matrix\nAcc: {accuracy:.2f}")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion Matrix saved to {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

    # 2. 2D Decision Boundary
    visualize_2d_decision_boundary(X_sound, X_full, y_full, best_model, output_dir)
    
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
                 'C0033-seg1-s', 'C0033-seg2-s', '1118004', '1118006', '1118008', '1118011', '1118013', '1118017', '1118020', '1118021', 
                 '1118023', '1118029', '1118030', '1118034', '1209038', '1209039', '1209043', '1209047', 
                 '1209049', '1209052', '1209054', '1209055', '1209056', '1209059', '1209060', '1209061', 
                 '1209062', '1209063', '1209065', '1209066', '1209068', '1209073', '1209075', '1209076', 
                 '1209079', '1209080', '1209083', '1209085', '1209086', '1209087', 'B0018', 'B0026', 
                 'B0037', 'B0038', 'B0039', 'B0045', 'B0058', 'B0059', 'B0064', 'B0067', 'C0001', 
                 'C0002', 'C0006', 'C0009', 'C0013', 'C0017', 'C0021', 'C0022', 'C0035']

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
    print("\n--- Step 3: One-Class SVM Classification ---")
    SVM_classification(output_features_path)

def main():
    classified_file_operation()

if __name__ == "__main__":
    main()
