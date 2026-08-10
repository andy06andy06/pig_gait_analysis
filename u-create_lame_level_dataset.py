#!/usr/bin/env python3
"""Create a 3-level lame dataset from pressure-mat symmetry tables.

This script reads ``videos/*_pressuremat.json`` files, computes the worst
same-limb-pair left/right asymmetry score from the ``Left Front / Right Front``
and ``Left Hind / Right Hind`` sections in each symmetry table, and copies the
matching DeepLabCut ``shuffle10`` h5 file into a new classified dataset folder.

The classification uses multiple pressure-mat symmetry metrics:

    Stance Time, Stride Time, Stride Length, Stride Velocity, Max Force

For each metric, the asymmetry score treats ratios above and below 1
symmetrically:

    score = max(ratio, 1 / ratio)

For each metric, the script first finds the worst front/hind left-right
asymmetry. The final per-pig score is a weighted average across metrics, with
Max Force intentionally weighted higher than the temporal/spatial metrics.
Default thresholds are:

    level1_sound:  weighted_asymmetry < 1.30
    level2_medium: 1.30 <= weighted_asymmetry < 1.50
    level3_lame:   weighted_asymmetry >= 1.50

Only h5 files containing ``shuffle10`` in the filename are copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_LEVEL1_MAX = 1.30
DEFAULT_LEVEL3_MIN = 1.50
SYMMETRY_SECTIONS = ("Left Front / Right Front", "Left Hind / Right Hind")
SYMMETRY_METRICS = (
    "Stance Time",
    "Stride Time",
    "Stride Length",
    "Stride Velocity",
    "Max Force",
)
METRIC_WEIGHTS = {
    "Stance Time": 0.05,
    "Stride Time": 0.05,
    "Stride Length": 0.05,
    "Stride Velocity": 0.05,
    "Max Force": 0.80,
}


def asymmetry_score(value: float) -> Optional[float]:
    """Return a direction-independent asymmetry score for a ratio."""
    if value == 0:
        return None
    return max(value, 1.0 / value)


def iter_symmetry_values(data: Dict[str, Any]) -> Iterable[Tuple[str, str, float, float]]:
    """Yield (section, metric, raw_ratio, score) for selected symmetry values."""
    sections = data.get("symmetry_table", {}).get("sections", {})
    for section_name in SYMMETRY_SECTIONS:
        metrics = sections.get(section_name, {})
        if not isinstance(metrics, dict):
            continue
        for metric_name in SYMMETRY_METRICS:
            raw_value = metrics.get(metric_name)
            if not isinstance(raw_value, (int, float)):
                continue
            score = asymmetry_score(float(raw_value))
            if score is None:
                continue
            yield section_name, metric_name, float(raw_value), score


def calculate_weighted_asymmetry(
    values: Iterable[Tuple[str, str, float, float]]
) -> Tuple[float, str, str, float, float, Dict[str, float]]:
    """Calculate weighted asymmetry after taking the worst score per metric.

    Returns final_score, dominant_section, dominant_metric, dominant_raw_ratio,
    dominant_unweighted_score, and per-metric worst scores. The dominant metric
    is selected by weighted contribution, so Max Force is prioritized according
    to METRIC_WEIGHTS.
    """
    worst_by_metric: Dict[str, Tuple[str, str, float, float]] = {}
    for section_name, metric_name, raw_ratio, score in values:
        previous = worst_by_metric.get(metric_name)
        if previous is None or score > previous[3]:
            worst_by_metric[metric_name] = (section_name, metric_name, raw_ratio, score)

    for metric in SYMMETRY_METRICS:
        # Missing pressure-mat symmetry values are treated as neutral/asymmetry-free
        # so incomplete metrics do not remove an otherwise usable pig sample.
        if metric not in worst_by_metric:
            worst_by_metric[metric] = ("missing", metric, 1.0, 1.0)

    weight_sum = sum(METRIC_WEIGHTS[metric] for metric in SYMMETRY_METRICS)
    if weight_sum <= 0:
        raise RuntimeError("Metric weights must sum to a positive value")

    final_score = sum(
        METRIC_WEIGHTS[metric] * worst_by_metric[metric][3] for metric in SYMMETRY_METRICS
    ) / weight_sum
    dominant = max(
        worst_by_metric.values(),
        key=lambda item: METRIC_WEIGHTS[item[1]] * item[3],
    )
    per_metric_scores = {
        metric: round(worst_by_metric[metric][3], 6) for metric in SYMMETRY_METRICS
    }
    return final_score, dominant[0], dominant[1], dominant[2], dominant[3], per_metric_scores


def classify(score: float, level1_max: float, level3_min: float) -> str:
    """Map a worst asymmetry score to the requested lame level."""
    if score < level1_max:
        return "level1_sound"
    if score < level3_min:
        return "level2_medium"
    return "level3_lame"


def infer_higher_side(section_name: str, raw_ratio: float) -> str:
    """Infer which side is higher from a Left / Right style ratio."""
    left_label, right_label = [part.strip() for part in section_name.split("/")]
    if raw_ratio > 1:
        return f"{left_label} higher than {right_label}"
    if raw_ratio < 1:
        return f"{right_label} higher than {left_label}"
    return f"{left_label} equal to {right_label}"


def find_shuffle10_h5(videos_dir: Path, pressuremat_id: str) -> Path:
    """Find the unique shuffle10 h5 file corresponding to one pressuremat id."""
    matches = sorted(videos_dir.glob(f"{pressuremat_id}*shuffle10*_filtered.h5"))
    if not matches:
        raise FileNotFoundError(f"No shuffle10 filtered h5 found for {pressuremat_id}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one shuffle10 h5 for {pressuremat_id}, found {len(matches)}: "
            + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def reset_output_dirs(output_dir: Path) -> None:
    """Create a clean classified dataset output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for level in ("level1_sound", "level2_medium", "level3_lame"):
        (output_dir / level).mkdir(parents=True, exist_ok=True)


def build_dataset(
    videos_dir: Path,
    output_dir: Path,
    level1_max: float,
    level3_min: float,
) -> List[Dict[str, Any]]:
    """Classify pressuremat ids and copy their shuffle10 h5 files."""
    if level1_max >= level3_min:
        raise ValueError("level1_max must be smaller than level3_min")

    reset_output_dirs(output_dir)
    summary: List[Dict[str, Any]] = []

    pressuremat_files = sorted(videos_dir.glob("*_pressuremat.json"))
    if not pressuremat_files:
        raise FileNotFoundError(f"No *_pressuremat.json files found in {videos_dir}")

    for pressuremat_path in pressuremat_files:
        pressuremat_id = pressuremat_path.name.replace("_pressuremat.json", "")
        data = json.loads(pressuremat_path.read_text(encoding="utf-8"))
        values = list(iter_symmetry_values(data))
        if not values:
            raise RuntimeError(f"No usable symmetry values found in {pressuremat_path}")

        (
            weighted_score,
            worst_section,
            worst_metric,
            worst_raw_value,
            worst_unweighted_score,
            per_metric_scores,
        ) = calculate_weighted_asymmetry(values)
        level = classify(weighted_score, level1_max, level3_min)
        h5_path = find_shuffle10_h5(videos_dir, pressuremat_id)
        destination = output_dir / level / h5_path.name
        shutil.copy2(h5_path, destination)

        summary.append(
            {
                "pressuremat_id": pressuremat_id,
                "level": level,
                "weighted_asymmetry_score": round(weighted_score, 6),
                "dominant_unweighted_score": round(worst_unweighted_score, 6),
                "dominant_section": worst_section,
                "dominant_metric": worst_metric,
                "dominant_raw_ratio": worst_raw_value,
                "per_metric_worst_scores": per_metric_scores,
                "higher_side_interpretation": infer_higher_side(worst_section, worst_raw_value),
                "source_pressuremat_json": str(pressuremat_path),
                "source_h5": str(h5_path),
                "copied_h5": str(destination),
            }
        )

    summary.sort(key=lambda row: (row["level"], row["pressuremat_id"]))
    write_reports(output_dir, summary, level1_max, level3_min)
    return summary


def write_reports(
    output_dir: Path,
    summary: List[Dict[str, Any]],
    level1_max: float,
    level3_min: float,
) -> None:
    """Write JSON and CSV classification reports."""
    counts = {"level1_sound": 0, "level2_medium": 0, "level3_lame": 0}
    for row in summary:
        counts[row["level"]] += 1

    report = {
        "settings": {
            "level1_sound": f"weighted_asymmetry < {level1_max}",
            "level2_medium": f"{level1_max} <= weighted_asymmetry < {level3_min}",
            "level3_lame": f"weighted_asymmetry >= {level3_min}",
            "score_method": "weighted average of per-metric worst same-limb-pair left/right asymmetry",
            "missing_metric_policy": "missing metric values are treated as neutral score 1.0",
            "symmetry_sections_used": list(SYMMETRY_SECTIONS),
            "symmetry_metrics_used": list(SYMMETRY_METRICS),
            "metric_weights": METRIC_WEIGHTS,
            "h5_selection": "shuffle10 only",
        },
        "counts": counts,
        "total": len(summary),
        "items": summary,
    }

    (output_dir / "classification_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "classification_summary.csv"
    fieldnames = list(summary[0].keys()) if summary else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default="videos", type=Path)
    parser.add_argument(
        "--output-dir", default=Path("videos/classified_video_new"), type=Path
    )
    parser.add_argument("--level1-max", default=DEFAULT_LEVEL1_MAX, type=float)
    parser.add_argument("--level3-min", default=DEFAULT_LEVEL3_MIN, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        level1_max=args.level1_max,
        level3_min=args.level3_min,
    )
    counts = {"level1_sound": 0, "level2_medium": 0, "level3_lame": 0}
    for row in summary:
        counts[row["level"]] += 1
    print(f"Created dataset at: {args.output_dir}")
    print(f"Total shuffle10 h5 files copied: {len(summary)}")
    for level, count in counts.items():
        print(f"  {level}: {count}")
    print(f"Reports written to: {args.output_dir / 'classification_summary.json'}")
    print(f"                    {args.output_dir / 'classification_summary.csv'}")


if __name__ == "__main__":
    main()