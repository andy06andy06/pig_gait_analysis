import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json


def _minimal_unique_prefixes(stems, min_len=5):
    """Return a mapping stem -> shortest unique prefix (>= min_len).

    - Starts from min_len characters for each name (or the full stem if shorter).
    - For any collisions, only the colliding names get lengthened step-by-step
      until all prefixes are unique (or equal to full stem).
    - If, after exhausting full lengths, collisions remain (identical stems),
      add a short numeric suffix to disambiguate deterministically.
    """
    stems = list(stems)
    # Initial desired lengths: at least min_len, but not longer than the stem itself
    lengths = {s: min(max(min_len, 1), len(s)) for s in stems}

    while True:
        buckets = {}
        for s in stems:
            k = s[:lengths[s]]
            buckets.setdefault(k, []).append(s)

        # Find any collisions (prefix shared by multiple stems)
        collisions = {k: v for k, v in buckets.items() if len(v) > 1}
        if not collisions:
            # All unique
            return {s: s[:lengths[s]] for s in stems}

        progressed = False
        for _, group in collisions.items():
            for s in group:
                if lengths[s] < len(s):
                    lengths[s] += 1
                    progressed = True

        if progressed:
            # Try again after increasing lengths for colliders
            continue

        # No further progress possible using prefixes only (identical stems).
        # Assign deterministic numeric suffixes.
        result = {s: s[:lengths[s]] for s in stems}
        seen = {}
        for s in stems:
            key = result[s]
            seen.setdefault(key, 0)
            cnt = seen[key]
            if cnt > 0:
                result[s] = f"{key}~{cnt}"
            seen[key] = cnt + 1
        return result

def read_h5_folder(folder_path):
    h5_arrays = []
    h5_filenames_arrays = []
    files = os.listdir(folder_path)
    h5_files = sorted([f for f in files if f.endswith('shuffle10_100000_filtered.h5')])
    if not h5_files:
        raise ValueError("No .h5 files found in the folder.")

    # Build minimal unique prefixes across all stems to avoid collisions
    stems = [os.path.splitext(f)[0] for f in h5_files]
    short_map = _minimal_unique_prefixes(stems, min_len=5)
    stem_by_file = {f: os.path.splitext(f)[0] for f in h5_files}

    for filename in h5_files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_hdf(file_path)
        h5_arrays.append(df)
        short_key = short_map[stem_by_file[filename]]
        h5_filenames_arrays.append(short_key)
    return h5_arrays, h5_filenames_arrays

def _save_pretty_json(path, data, indent=4, sort_keys=True):
    """Write JSON with 4-space indent at top level and compact inner values.

    Produces one line per top-level (per-video) entry, e.g.:
    {
        "00001": {"FL":[...],"FR":[...],"BL":[...],"BR":[...]},
        "00002": {...}
    }
    """
    if not isinstance(data, dict):
        # Fallback to normal pretty dump if not a dict
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
        return

    items = data.items()
    if sort_keys:
        items = sorted(items, key=lambda kv: str(kv[0]))

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        count = 0
        total = len(list(items))
        # Need to re-materialize items after len()
        if sort_keys:
            items = sorted(data.items(), key=lambda kv: str(kv[0]))
        else:
            items = list(data.items())
        for idx, (k, v) in enumerate(items):
            f.write(" ")
            f.write(" " * (indent - 1))  # total 4 spaces in front of each key
            f.write(json.dumps(k, ensure_ascii=False))
            f.write(": ")
            # compact inner structure for the value
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if idx < len(items) - 1:
                # comma + newline + blank line between videos
                f.write(",\n\n")
            else:
                f.write("\n")
        f.write("}\n")


def _save_stage_snapshot_plots(stage_series, stage_order, starts_only, subfolder, short_name):
    """Save per-stage 2x2 leg plots and one combined grid showing all stages."""
    leg_order = ["FL", "FR", "BL", "BR"]
    leg_colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "BL": "#2ca02c", "BR": "#d62728"}
    stage_dir = os.path.join(subfolder, "stages")
    os.makedirs(stage_dir, exist_ok=True)

    # Individual stage plots (2x2 legs)
    for stage in stage_order:
        data = stage_series.get(stage, {})
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False, sharey=False)
        for idx, leg in enumerate(leg_order):
            ax = axes[idx // 2, idx % 2]
            series = data.get(leg)
            if series is None:
                ax.set_title(f"{leg}: missing")
                continue
            ax.plot(series.index, series.values, lw=0.8, color=leg_colors.get(leg, "tab:blue"), label=f"{leg} x")
            if stage == stage_order[-1]:
                for kf in starts_only.get(leg, []):
                    ax.axvline(kf, color="red", linestyle="--", linewidth=0.8)
            ax.set_title(leg)
            ax.set_xlabel("frame")
            ax.set_ylabel("x (px)")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        fig.suptitle(f"{short_name} - {stage}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(stage_dir, f"stage_{stage}_{short_name}.png"), dpi=300)
        plt.close(fig)

    # Combined grid: rows=legs, cols=stages
    n_rows = len(leg_order)
    n_cols = len(stage_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=False, sharey=False)
    stage_labels = [f"Stage {i+1}" for i in range(n_cols)]
    for r, leg in enumerate(leg_order):
        for c, stage in enumerate(stage_order):
            ax = axes[r, c]
            series = stage_series.get(stage, {}).get(leg)
            if series is not None:
                ax.plot(series.index, series.values, lw=0.8, color=leg_colors.get(leg, "tab:blue"))
                if stage == stage_order[-1]:
                    for kf in starts_only.get(leg, []):
                        ax.axvline(kf, color="red", linestyle="--", linewidth=0.8)
            ax.set_title(f"{leg} | {stage_labels[c]}")
            ax.set_xlabel("frame")
            ax.set_ylabel("x (px)")
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    arrow_flow = " \u2192 ".join(stage_labels)
    fig.suptitle(f"{short_name} - {arrow_flow}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(stage_dir, f"stages_combined_{short_name}.png"), dpi=300)
    plt.close(fig)

def plot_frame_vs_x(h5_arrays, h5_filenames_arrays, folder_path):

    for i, df in enumerate(h5_arrays):
        # 建立每個 h5 檔案對應的資料夾
        subfolder = os.path.join(folder_path, h5_filenames_arrays[i])
        os.makedirs(subfolder, exist_ok=True)

        # 展平 MultiIndex 欄位（DLC 格式）
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

        FL_hoof_x = df.columns[0]
        FL_hoof_y = df.columns[1]
        FL_hoof_likelihood = df.columns[2]
        FL_knee_x = df.columns[3]
        FL_knee_y = df.columns[4]
        FL_knee_likelihood = df.columns[5]
        FR_hoof_x = df.columns[6]
        FR_hoof_y = df.columns[7]
        FR_hoof_likelihood = df.columns[8]
        FR_knee_x = df.columns[9]
        FR_knee_y = df.columns[10]
        FR_knee_likelihood = df.columns[11]
        BL_hoof_x = df.columns[12]
        BL_hoof_y = df.columns[13]
        BL_hoof_likelihood = df.columns[14]
        BL_knee_x = df.columns[15]
        BL_knee_y = df.columns[16]
        BL_knee_likelihood = df.columns[17]
        BR_hoof_x = df.columns[18]
        BR_hoof_y = df.columns[19]
        BR_hoof_likelihood = df.columns[20]
        BR_knee_x = df.columns[21]
        BR_knee_y = df.columns[22]
        BR_knee_likelihood = df.columns[23]
        F_elbow_x = df.columns[24]
        F_elbow_y = df.columns[25]
        F_elbow_likelihood = df.columns[26]
        B_elbow_x = df.columns[27]
        B_elbow_y = df.columns[28]
        B_elbow_likelihood = df.columns[29]
        F_back_x = df.columns[30]
        F_back_y = df.columns[31]
        F_back_likelihood = df.columns[32]
        M_back_x = df.columns[33]
        M_back_y = df.columns[34]
        M_back_likelihood = df.columns[35]
        B_back_x = df.columns[36]
        B_back_y = df.columns[37]
        B_back_likelihood = df.columns[38]
        nose_x = df.columns[39]
        nose_y = df.columns[40]
        nose_likelihood = df.columns[41]
        F_improvement_x = df.columns[42]
        F_improvement_y = df.columns[43]
        F_improvement_likelihood = df.columns[44]
        B_improvement_x = df.columns[45]
        B_improvement_y = df.columns[46]
        B_improvement_likelihood = df.columns[47]

        leg_col_map = {
            "FL": FL_hoof_x,
            "FR": FR_hoof_x,
            "BL": BL_hoof_x,
            "BR": BR_hoof_x,
        }
        leg_colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "BL": "#2ca02c", "BR": "#d62728"}
        stage_order = [
            "01_original",
            "02_outliers_removed",
            "03_short_removed",
            "04_nan_merged",
            "05_close_merged",
            "06_final",
        ]
        stage_series = {}

        def _snapshot(stage_name):
            stage_series[stage_name] = {
                leg: df[col].copy()
                for leg, col in leg_col_map.items()
                if col in df.columns
            }
        _snapshot("01_original")
        # Remove outliers by checking the difference between consecutive frames
        remove_outliers(df, threshold=5, target_columns=list(leg_col_map.values()))
        _snapshot("02_outliers_removed")
        # Remove short-lasting points
        remove_short_lasting_points(df, min_length=20, target_columns=list(leg_col_map.values()))
        _snapshot("03_short_removed")
        # Merge small NaN gaps
        merge_small_nan_gaps(df, max_nan_gap=5, target_columns=list(leg_col_map.values()), fill_method="prev")
        _snapshot("04_nan_merged")
        # Merge close points
        merge_close_points(df, max_distance=30, target_columns=list(leg_col_map.values()))
        _snapshot("05_close_merged")
        
        # Detect the keyframe
        segments, starts_only = detect_keyframe(df, delta_px=6.0, min_len=20, p_cutoff=None)
        _snapshot("06_final")
        _save_stage_snapshot_plots(stage_series, stage_order, starts_only, subfolder, h5_filenames_arrays[i])
        # 或使用 segments['FL'] 取得每段的 start/end/mean_x/length

        # 以兩個獨立檔案存放：starts_only 與 segments（不覆蓋整個檔案）
        base_dir = os.path.dirname(__file__)
        starts_path = os.path.join(base_dir, "1-keyframes_starts_only.json")
        segments_path = os.path.join(base_dir, "1-keyframes_segments.json")

        def _load_dict(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                return data
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

        video_key = str(h5_filenames_arrays[i])
        legacy_keys = {"FL", "FR", "BL", "BR"}

        # starts_only：若檔案是舊平坦格式，包成當前影片的 key
        existing_starts = _load_dict(starts_path)
        if existing_starts and set(existing_starts.keys()) == legacy_keys:
            existing_starts = {video_key: existing_starts}
        existing_starts[video_key] = starts_only
        _save_pretty_json(starts_path, existing_starts, indent=4, sort_keys=True)
        print(f"Saved starts_only for {video_key} -> {starts_path}")

        # segments：直接合併寫入
        existing_segments = _load_dict(segments_path)
        existing_segments[video_key] = segments
        _save_pretty_json(segments_path, existing_segments, indent=4, sort_keys=True)
        print(f"Saved segments for {video_key} -> {segments_path}")

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[FL_hoof_x], label=FL_hoof_x.split('100000_')[1], color=leg_colors["FL"])
        plt.plot(df.index, df[FR_hoof_x], label=FR_hoof_x.split('100000_')[1], color=leg_colors["FR"])
        plt.plot(df.index, df[BL_hoof_x], label=BL_hoof_x.split('100000_')[1], color=leg_colors["BL"])
        plt.plot(df.index, df[BR_hoof_x], label=BR_hoof_x.split('100000_')[1], color=leg_colors["BR"])
        plt.xlabel("frame")
        plt.ylabel("keypoint x position (px)")
        plt.title(f'Plot of {h5_filenames_arrays[i]}')
        plt.legend()
        plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        save_path = os.path.join(subfolder, f'x_vs_frame_{h5_filenames_arrays[i]}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot to: {save_path}")

def remove_outliers(df, threshold, target_columns=None):
    if target_columns is None:
        target_columns = df.columns
    for col in target_columns:
        x = df[col].values
        diff = np.abs(np.diff(x, prepend=x[0]))
        outlier_idx = np.where(diff > threshold)[0]
        df.loc[outlier_idx, col] = np.nan

def remove_short_lasting_points(df, min_length, target_columns=None):
    if target_columns is None:
        target_columns = df.columns
    for col in target_columns:
        mask = ~df[col].isna()
        group = (mask != mask.shift()).cumsum()
        lengths = mask.groupby(group).transform('sum')
        short_mask = (mask & (lengths < min_length))
        df.loc[short_mask, col] = np.nan

def merge_close_points(df, max_distance=15, target_columns=None):
    if target_columns is None:
        target_columns = df.columns

    for col in target_columns:
        series = df[col]
        is_valid = ~series.isna()

        # Step 1: 標記連續非NaN區段
        group_id = (is_valid != is_valid.shift()).cumsum()
        groups = []
        current_group = []

        for idx, valid in enumerate(is_valid):
            if valid:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        if current_group:
            groups.append(current_group)

        # Step 2: 合併相鄰平均差距小於閾值的群組
        merged_series = series.copy()
        merged_groups = []
        i = 0
        while i < len(groups):
            current_group = groups[i]
            j = i + 1

            while j < len(groups):
                avg_current = np.nanmean(series.iloc[current_group])
                avg_next = np.nanmean(series.iloc[groups[j]])
                if abs(avg_current - avg_next) < max_distance:
                    # 合併 current_group 與下一個
                    current_group = current_group + groups[j]
                    j += 1
                else:
                    break
            # 將合併後的群組寫回平均值
            avg_val = np.nanmean(series.iloc[current_group])
            merged_series.iloc[current_group] = avg_val
            merged_groups.append(current_group)
            i = j  # 跳到下一段未合併的 group

        df[col] = merged_series

def merge_small_nan_gaps(df, max_nan_gap=5, target_columns=None, fill_method="prev"):
    if target_columns is None:
        target_columns = df.columns

    for col in target_columns:
        s = df[col].copy()
        is_nan = s.isna().to_numpy()

        n = len(s)
        i = 0
        while i < n:
            if not is_nan[i]:
                i += 1
                continue

            # 找一段連續 NaN [i, j)
            j = i
            while j < n and is_nan[j]:
                j += 1
            gap_len = j - i  # NaN 區間長度

            if 1 <= gap_len <= max_nan_gap:
                # 檢查前面是否有有效值區段
                prev_idx = i - 1
                if prev_idx >= 0 and not is_nan[prev_idx]:
                    # 取得「前一段有效區間」的索引範圍
                    # 往前找到前段的開頭
                    k = prev_idx
                    while k - 1 >= 0 and not is_nan[k - 1]:
                        k -= 1
                    prev_segment_idx = np.arange(k, prev_idx + 1)

                    if fill_method == "prev":
                        fill_val = s.iloc[prev_idx]
                    elif fill_method == "mean":
                        fill_val = float(np.nanmean(s.iloc[prev_segment_idx]))
                    else:
                        raise ValueError("fill_method must be 'prev' or 'mean'")

                    # 將這段短 NaN 用前段代表值補起來
                    s.iloc[i:j] = fill_val
                    is_nan[i:j] = False  # 更新狀態，後續掃描可無縫接續
                # 若前面沒有有效段，就不補

            # 移動到下一段
            i = j

        df[col] = s

def detect_keyframe(df, delta_px=6.0, min_len=3, p_cutoff=None):
    """
    Detect hoof keyframes using swing/stance from frame-to-frame horizontal displacement.
    A frame is stance if |Δx| < delta_px; stance segments are merged when their mean x
    differs by less than delta_px. Edge segments are dropped to keep complete strides.

    Parameters
    ----------
    df : pandas.DataFrame
        A DLC predictions dataframe with flattened columns.
    delta_px : float
        Displacement threshold for swing/stance and merging adjacent stance segments.
    min_len : int
        Minimum number of consecutive stance frames to keep a segment.
    p_cutoff : float or None
        If provided and a corresponding likelihood column exists, frames with likelihood
        below this cutoff are treated as gaps (ignored for segment continuity).

    Returns
    -------
    result : dict
        {
          'FL': [{'start': int, 'end': int, 'mean_x': float, 'length': int}, ...],
          'FR': [...],
          'BL': [...],
          'BR': [...]
        }
    starts_only : dict
        {'FL': [start_idx, ...], 'FR': [...], 'BL': [...], 'BR': [...]}  # convenience
    """

    # Helper: find a column by fuzzy substring match (works with DLC auto names)
    def _find_col(sub):
        matches = [c for c in df.columns if sub in c]
        if not matches:
            return None
        # Prefer the first exact-ish match; otherwise return the first
        matches.sort(key=lambda c: len(c))
        return matches[0]

    # Try to locate hoof x and likelihood columns by substring
    col_map = {
        'FL': dict(x=_find_col('FL_hoof_x'), like=_find_col('FL_hoof_likelihood')),
        'FR': dict(x=_find_col('FR_hoof_x'), like=_find_col('FR_hoof_likelihood')),
        'BL': dict(x=_find_col('BL_hoof_x'), like=_find_col('BL_hoof_likelihood')),
        'BR': dict(x=_find_col('BR_hoof_x'), like=_find_col('BR_hoof_likelihood')),
    }

    def _segment_series(x_series, like_series=None):
        xv = x_series.to_numpy(dtype=float) 
        n = len(xv)

        # Apply likelihood cutoff as gaps (NaN) if available
        if like_series is not None and p_cutoff is not None:
            likev = like_series.to_numpy(dtype=float)
            xv = np.where(likev >= float(p_cutoff), xv, np.nan)

        # Frame-to-frame displacement |Δx|
        dx = np.full(n, np.nan, dtype=float)
        for idx in range(1, n):
            if not np.isnan(xv[idx]) and not np.isnan(xv[idx - 1]):
                dx[idx] = abs(xv[idx] - xv[idx - 1])

        stance_mask = (dx < float(delta_px))  # True means stance; NaN stays False

        # Extract stance segments
        raw_segments = []
        i = 0
        while i < n:
            while i < n and not (stance_mask[i] and not np.isnan(xv[i])):
                i += 1
            if i >= n:
                break
            start = i
            j = i + 1
            while j < n and stance_mask[j] and not np.isnan(xv[j]):
                j += 1
            end = j - 1
            length = end - start + 1
            if length >= int(min_len):
                raw_segments.append({
                    'start': int(start),
                    'end': int(end),
                    'mean_x': float(np.nanmean(xv[start:end+1])),
                    'length': int(length),
                })
            i = j

        # Merge adjacent stance segments if mean x differs by less than delta_px
        if not raw_segments:
            return []
        merged = [raw_segments[0]]
        for seg in raw_segments[1:]:
            prev = merged[-1]
            if abs(seg['mean_x'] - prev['mean_x']) < float(delta_px):
                total_len = prev['length'] + seg['length']
                weighted_mean = (
                    prev['mean_x'] * prev['length'] + seg['mean_x'] * seg['length']
                ) / total_len
                merged[-1] = {
                    'start': prev['start'],
                    'end': seg['end'],
                    'mean_x': float(weighted_mean),
                    'length': int(total_len),
                }
            else:
                merged.append(seg)

        # Drop incomplete edge segments (beginning and end of the video)
        if len(merged) <= 2:
            return []
        return merged[1:-1]

    result = {}
    for leg, cols in col_map.items():
        xcol = cols['x']
        if xcol is None or xcol not in df.columns:
            # if not found, return empty list for that leg
            result[leg] = []
            continue
        likecol = cols['like'] if cols['like'] in df.columns else None
        segs = _segment_series(df[xcol], df[likecol] if likecol else None)
        result[leg] = segs

    starts_only = {leg: [s['start'] for s in segs] for leg, segs in result.items()}
    return result, starts_only


def find_irregular_tempo_files(folder_path, cv_threshold=0.2):
    """
    Identify files where feet do not have a regular tempo of stance period.
    Uses Coefficient of Variation (CV) of stride duration (start-to-start interval).
    Checks filtered h5 files after applying the same cleaning as in plot_frame_vs_x.
    """
    irregular_files = []
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return []
        
    h5_files = sorted([f for f in files if f.endswith('shuffle10_100000_filtered.h5')])

    for filename in h5_files:
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_hdf(file_path)
            # Flatten columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Identify columns for cleaning
            target_cols = []
            for leg in ['FL', 'FR', 'BL', 'BR']:
                 # Find hoof_x column
                 matches = [c for c in df.columns if f'{leg}_hoof_x' in c]
                 if matches:
                     target_cols.append(matches[0])
            
            if not target_cols:
                continue
            
            # Apply cleaning (same as in plot_frame_vs_x)
            remove_outliers(df, threshold=5, target_columns=target_cols)
            remove_short_lasting_points(df, min_length=20, target_columns=target_cols)
            merge_small_nan_gaps(df, max_nan_gap=5, target_columns=target_cols, fill_method="prev")
            merge_close_points(df, max_distance=30, target_columns=target_cols)
            
            # Detect keyframes
            segments, _ = detect_keyframe(df, delta_px=6.0, min_len=20, p_cutoff=None)
            
            file_irregular = False
            for leg, segs in segments.items():
                if len(segs) < 3:
                     # Skip if too few segments to judge regularity
                     continue
                
                # Calculate stride times (start to start)
                starts = [s['start'] for s in segs]
                strides = np.diff(starts)
                
                if len(strides) < 2:
                    continue
                    
                mean_stride = np.mean(strides)
                std_stride = np.std(strides)
                cv = std_stride / mean_stride if mean_stride > 0 else 0
                
                if cv > cv_threshold:
                    file_irregular = True
                    break
            
            if file_irregular:
                irregular_files.append(filename)

        except Exception as e:
            print(f"Error checking tempo for {filename}: {e}")
            
    return irregular_files

h5_arrays, h5_filenames_arrays = read_h5_folder('../videos')

irregular_tempo_files = find_irregular_tempo_files('../videos', cv_threshold=0.2)
print("Files with irregular tempo (CV > 0.2):", irregular_tempo_files)

plot_frame_vs_x(h5_arrays, h5_filenames_arrays, '../videos/plots')
