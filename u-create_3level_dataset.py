#!/usr/bin/env python3
"""
Create a 3-level lame dataset (Level 0: Sound, Level 1: Medium, Level 2: Severe)
from all 91 videos in videos/ directory.

Level 0 (Sound / 健康): 58 videos
Level 1 (Medium Lameness / 輕中度跛行): 31 videos (12 visual/manual + 19 objective pressure-mat asymmetry >= 1.45)
Level 2 (Severe Lameness / 最嚴重跛行): 2 videos ('LsideLHlame,level3', '0627,Y1804-7-seg3')
"""

import os
import glob
import json
import csv
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR.parent / "videos"
OUTPUT_DIR = VIDEOS_DIR / "classified_video_3levels"
FEATURE_JSON_PATH = BASE_DIR / "3-keyframe_features.json"
OUTPUT_FEATURE_PATH = BASE_DIR / "6-classified_3level_features.json"

ASYMMETRY_THRESHOLD = 1.45

LEVEL2_IDS = {
    "LsideLHlame,level3",
    "0627,Y1804-7-seg3",
}

LEVEL1_IDS = {
    "C0033-seg1-lameness",
    "C0033-seg2-lameness",
    "C0038-seg1-lameness",
    "C0038-seg2-lameness",
    "1209045",
    "C0004",
    "C0008",
    "C0014",
    "C0015",
    "C0019",
    "C0020",
    "C0040",
}

LEVEL_MAPPING = {
    "level0_sound": "Level 0 (健康 / Sound)",
    "level1_medium": "Level 1 (輕中度跛行 / Medium Lameness)",
    "level2_severe": "Level 2 (最嚴重跛行 / Severe Lameness)",
}

def load_pressuremat_scores(videos_dir: Path) -> dict:
    """Load max force asymmetry scores from videos/*_pressuremat.json."""
    scores = {}
    for p in videos_dir.glob("*_pressuremat.json"):
        vid = p.name.replace("_pressuremat.json", "")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            sec = data.get("symmetry_table", {}).get("sections", {})
            lf_rf = sec.get("Left Front / Right Front", {}).get("Max Force")
            lh_rh = sec.get("Left Hind / Right Hind", {}).get("Max Force")
            score = max(max(lf_rf, 1.0 / lf_rf) if lf_rf else 1.0, max(lh_rh, 1.0 / lh_rh) if lh_rh else 1.0)
            scores[vid] = round(float(score), 4)
        except Exception as e:
            print(f"Warning: Failed to parse pressure mat for {vid}: {e}")
    return scores

def parse_video_id(filename: str) -> str:
    base = os.path.basename(filename)
    if "DLC_" in base:
        return base.split("DLC_", 1)[0]
    return os.path.splitext(base)[0]

def map_vid_to_feat_key(vid: str, feat_keys: set) -> str | None:
    if vid in feat_keys:
        return vid
    if vid == "0627,Y1804-7-seg3":
        return "0627,"
    if vid == "LsideLHlame,level3":
        return "Lside"
    if f"{vid}D" in feat_keys:
        return f"{vid}D"
    if vid.endswith("-sound"):
        candidate = vid[:-len("-sound")]
        if candidate in feat_keys:
            return candidate
        candidate_s = candidate + "-s"
        if candidate_s in feat_keys:
            return candidate_s
    if vid.endswith("-lameness"):
        candidate = vid[:-len("-lameness")]
        if candidate in feat_keys:
            return candidate
        candidate_l = candidate + "-l"
        if candidate_l in feat_keys:
            return candidate_l
    return None

def build_3level_dataset():
    h5_files = sorted(VIDEOS_DIR.glob("*shuffle10*_filtered.h5"))
    print(f"Found {len(h5_files)} shuffle10 filtered h5 files in {VIDEOS_DIR}")

    # Load pressure mat scores for objective biomechanical reassignment
    pressuremat_scores = load_pressuremat_scores(VIDEOS_DIR)
    print(f"Loaded {len(pressuremat_scores)} pressure mat asymmetry records (Threshold T >= {ASYMMETRY_THRESHOLD})")

    # Prepare directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for level in LEVEL_MAPPING:
        (OUTPUT_DIR / level).mkdir(parents=True, exist_ok=True)

    with open(FEATURE_JSON_PATH, "r", encoding="utf-8") as f:
        all_features = json.load(f)
    feat_keys = set(all_features.keys())

    summary = []
    classified_features = {level: {} for level in LEVEL_MAPPING}
    counts = {level: 0 for level in LEVEL_MAPPING}
    reassigned_from_sound = []

    for h5_path in h5_files:
        vid = parse_video_id(h5_path.name)
        mat_score = pressuremat_scores.get(vid, None)

        if vid in LEVEL2_IDS:
            level = "level2_severe"
            reason = "Clinical / Visual Severe Lameness"
        elif vid in LEVEL1_IDS:
            level = "level1_medium"
            reason = "Visual / Experimental Medium Lameness"
        elif mat_score is not None and mat_score >= ASYMMETRY_THRESHOLD:
            level = "level1_medium"
            reason = f"Pressure Mat Asymmetry {mat_score:.3f} >= {ASYMMETRY_THRESHOLD}"
            reassigned_from_sound.append((vid, mat_score))
        else:
            level = "level0_sound"
            if mat_score is not None:
                reason = f"Pressure Mat Asymmetry {mat_score:.3f} < {ASYMMETRY_THRESHOLD} (Sound)"
            else:
                reason = "Visual Healthy / Sound Locomotion"

        dest_h5 = OUTPUT_DIR / level / h5_path.name
        shutil.copy2(h5_path, dest_h5)
        counts[level] += 1

        feat_key = map_vid_to_feat_key(vid, feat_keys)
        if feat_key and feat_key in all_features:
            classified_features[level][vid] = all_features[feat_key]
        else:
            raise KeyError(f"Error: No feature key mapped for {vid}")

        summary.append({
            "video_id": vid,
            "level": level,
            "level_description": LEVEL_MAPPING[level],
            "pressuremat_asymmetry": mat_score,
            "assignment_reason": reason,
            "feature_key": feat_key,
            "source_h5": str(h5_path),
            "copied_h5": str(dest_h5),
        })

    # Save summary json & csv
    summary.sort(key=lambda x: (x["level"], x["video_id"]))
    report = {
        "settings": {
            "total_videos": len(summary),
            "level_mapping": LEVEL_MAPPING,
            "asymmetry_threshold": ASYMMETRY_THRESHOLD,
            "level2_ids": sorted(list(LEVEL2_IDS)),
            "level1_visual_ids": sorted(list(LEVEL1_IDS)),
            "level1_pressuremat_reassigned_ids": sorted([v[0] for v in reassigned_from_sound]),
        },
        "counts": counts,
        "items": summary,
    }

    with open(OUTPUT_DIR / "classification_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "classification_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with open(OUTPUT_FEATURE_PATH, "w", encoding="utf-8") as f:
        json.dump(classified_features, f, indent=4, ensure_ascii=False)

    print("\n==========================================")
    print(" 3-Level Dataset Creation Complete")
    print("==========================================")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Total Videos: {len(summary)}")
    for level, count in counts.items():
        print(f"  {LEVEL_MAPPING[level]}: {count} videos")
    print(f"  Reassigned from Sound to Medium via Pressure Mat (>= {ASYMMETRY_THRESHOLD}): {len(reassigned_from_sound)} videos")
    print(f"Feature JSON saved to: {OUTPUT_FEATURE_PATH}")
    print(f"Summary JSON saved to: {OUTPUT_DIR / 'classification_summary.json'}")
    print(f"Summary CSV saved to:  {OUTPUT_DIR / 'classification_summary.csv'}")

if __name__ == "__main__":
    build_3level_dataset()
