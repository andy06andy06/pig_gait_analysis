import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency in some envs
    cv2 = None

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
    return df

def _build_part_columns(columns):
    parts = {}
    for col in columns:
        if col.endswith("_x"):
            base = col[:-2]
            parts.setdefault(base, {})["x"] = col
        elif col.endswith("_y"):
            base = col[:-2]
            parts.setdefault(base, {})["y"] = col
        elif col.endswith("_likelihood"):
            base = col[: -len("_likelihood")]
            parts.setdefault(base, {})["likelihood"] = col
    return parts

def _find_h5_for_key(h5_dir: str, video_key: str) -> str | None:
    candidates = [f for f in os.listdir(h5_dir) if f.startswith(video_key) and f.endswith(".h5")]
    if not candidates:
        return None
    filtered = [f for f in candidates if "filtered" in f.lower()]
    chosen = filtered[0] if filtered else sorted(candidates)[0]
    return os.path.join(h5_dir, chosen)

def _find_video_for_key(video_dir: str, video_key: str) -> str | None:
    candidates = [f for f in os.listdir(video_dir) if f.startswith(video_key) and f.lower().endswith(".mp4")]
    if not candidates:
        return None
    return os.path.join(video_dir, sorted(candidates)[0])

def _save_pretty_top_level(path, data, indent=4):
    # One line per top-level key, compact inner JSON, blank line between items
    if not isinstance(data, dict):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=True)
        return
    items = sorted(data.items(), key=lambda kv: str(kv[0]))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        for idx, (k, v) in enumerate(items):
            f.write(" " * indent)
            f.write(json.dumps(k, ensure_ascii=False))
            f.write(": ")
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if idx < len(items) - 1:
                f.write(",\n\n")
            else:
                f.write("\n")
        f.write("}\n")

def _draw_keypoints_bgr(
    frame_bgr: np.ndarray,
    coords: Dict[str, Dict[str, Any]],
    radius: int = 4,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw keypoints on a BGR frame in-place and return it.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required to draw keypoints. Please install opencv-python.")

    for part, pt in coords.items():
        if not isinstance(pt, dict):
            continue
        x, y = pt.get("x"), pt.get("y")
        if x is None or y is None:
            continue
        try:
            px = int(round(float(x)))
            py = int(round(float(y)))
        except Exception:
            continue
        cv2.circle(frame_bgr, (px, py), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return frame_bgr

def _crop_zoom(
    frame_bgr: np.ndarray,
    coords: Dict[str, Dict[str, Any]],
    margin: int = 100,
    scale: float = 2.0,
) -> Optional[np.ndarray]:
    """Crop a tight box around keypoints (with margin) and upscale."""
    h, w = frame_bgr.shape[:2]
    xs: List[int] = []
    ys: List[int] = []
    for pt in coords.values():
        if not isinstance(pt, dict):
            continue
        x, y = pt.get("x"), pt.get("y")
        if x is None or y is None:
            continue
        try:
            xs.append(int(round(float(x))))
            ys.append(int(round(float(y))))
        except Exception:
            continue

    if not xs or not ys:
        return None

    x1 = max(min(xs) - margin, 0)
    x2 = min(max(xs) + margin, w)
    y1 = max(min(ys) - margin, 0)
    y2 = min(max(ys) + margin, h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop

def collect_keyframe_coords(starts_json_path: str, h5_dir: str, output_path: str | None = None,
                            how: str = "union"):
    """
    Read starts_only per video and collect coordinates for all body parts at each keyframe.

    - how: "union" (default) takes the union of frames across FL/FR/BL/BR.
            "by_leg" keeps frames grouped by leg.
    - output_path: if provided, writes JSON using the preferred formatting.
    """
    with open(starts_json_path, "r", encoding="utf-8") as f:
        starts_root = json.load(f)
    if not isinstance(starts_root, dict):
        raise ValueError("starts_json must be a dict mapping video_key -> starts_only")

    result = {}

    for video_key, payload in starts_root.items():
        # Payload can be starts_only dict or older flat dict
        if isinstance(payload, dict) and set(payload.keys()) >= {"FL", "FR", "BL", "BR"}:
            starts_only = payload
        elif isinstance(payload, dict) and "starts_only" in payload:
            starts_only = payload["starts_only"]
        else:
            # Skip unrecognized structure
            continue

        h5_path = _find_h5_for_key(h5_dir, str(video_key))
        if not h5_path or not os.path.isfile(h5_path):
            # No data file to read for this video
            continue

        df = pd.read_hdf(h5_path)
        df = _flatten_columns(df)
        parts = _build_part_columns(df.columns)

        def row_to_coords(idx: int):
            if idx < 0 or idx >= len(df):
                return None
            row = df.iloc[idx]
            coords = {}
            for base, cols in parts.items():
                x = row.get(cols.get("x"), np.nan)
                y = row.get(cols.get("y"), np.nan)
                like = row.get(cols.get("likelihood"), np.nan)
                short_base = base.split("100000_")[-1]
                coords[short_base] = {
                    "x": float(x) if pd.notna(x) else None,
                    "y": float(y) if pd.notna(y) else None,
                    "likelihood": float(like) if pd.notna(like) else None,
                }
            return coords

        if how == "union":
            frames = set()
            for leg in ("FL", "FR", "BL", "BR"):
                if leg in starts_only and isinstance(starts_only[leg], list):
                    frames.update(int(x) for x in starts_only[leg])
            keyframes = sorted(frames)
            items = [
                {"frame": int(fr), "coords": row_to_coords(int(fr))}
                for fr in keyframes if row_to_coords(int(fr)) is not None
            ]
            # Detect walking direction and store at the per-video top level (not per item)
            direction = None
            if detect_walk_direction is not None:
                try:
                    direction = detect_walk_direction(str(video_key), coords_data={str(video_key): items})
                except Exception:
                    direction = None
            # Wrap as an object with top-level direction and items list
            result[video_key] = {
                "direction": direction or "unknown",
                "items": items,
            }
        elif how == "by_leg":
            out = {}
            for leg in ("FL", "FR", "BL", "BR"):
                frames = starts_only.get(leg, []) or []
                out[leg] = [
                    {"frame": int(fr), "coords": row_to_coords(int(fr))}
                    for fr in frames if row_to_coords(int(fr)) is not None
                ]
            # Detect walking direction and store at top-level alongside legs
            if detect_walk_direction is not None:
                try:
                    direction = detect_walk_direction(str(video_key), coords_data={str(video_key): out})
                    out["direction"] = direction
                except Exception:
                    pass
            result[video_key] = out
        else:
            raise ValueError("how must be 'union' or 'by_leg'")

    if output_path:
        _save_pretty_top_level(output_path, result)
    return result

def save_keyframe_images(
    starts_json_path: str,
    h5_dir: str,
    video_dir: str,
    output_root: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Save keyframe images with predicted keypoints overlaid.

    - starts_json_path: path to keyframes_starts_only.json
    - h5_dir: directory containing DLC .h5 outputs
    - video_dir: directory containing source .mp4 videos
    - output_root: root folder to store images; defaults to <video_dir>/keyframe_image

    Returns dict mapping video_key -> list of saved frame indices.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required to export keyframe images. Please install opencv-python.")

    with open(starts_json_path, "r", encoding="utf-8") as f:
        starts_root = json.load(f)
    if not isinstance(starts_root, dict):
        raise ValueError("starts_json must be a dict mapping video_key -> starts_only")

    output_root = output_root or os.path.join(video_dir, "keyframe_image")
    os.makedirs(output_root, exist_ok=True)

    saved: Dict[str, List[int]] = {}

    for video_key, starts_only in starts_root.items():
        if not isinstance(starts_only, dict):
            continue

        h5_path = _find_h5_for_key(h5_dir, str(video_key))
        if not h5_path or not os.path.isfile(h5_path):
            print(f"[warn] Skip {video_key}: missing h5 predictions")
            continue

        video_path = _find_video_for_key(video_dir, str(video_key))
        if not video_path or not os.path.isfile(video_path):
            print(f"[warn] Skip {video_key}: missing video file")
            continue

        df = pd.read_hdf(h5_path)
        df = _flatten_columns(df)
        parts = _build_part_columns(df.columns)

        def row_to_coords(idx: int):
            if idx < 0 or idx >= len(df):
                return None
            row = df.iloc[idx]
            coords = {}
            for base, cols in parts.items():
                x = row.get(cols.get("x"), np.nan)
                y = row.get(cols.get("y"), np.nan)
                like = row.get(cols.get("likelihood"), np.nan)
                short_base = base.split("100000_")[-1]
                coords[short_base] = {
                    "x": float(x) if pd.notna(x) else None,
                    "y": float(y) if pd.notna(y) else None,
                    "likelihood": float(like) if pd.notna(like) else None,
                }
            return coords

        legs = ("FL", "FR", "BL", "BR")
        out_dir_root = os.path.join(output_root, str(video_key))
        os.makedirs(out_dir_root, exist_ok=True)

        per_video_saved: List[int] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[warn] Skip {video_key}: cannot open video")
            continue

        try:
            for leg in legs:
                frames = starts_only.get(leg, []) or []
                if not isinstance(frames, list) or not frames:
                    continue
                out_dir = os.path.join(out_dir_root, leg)
                os.makedirs(out_dir, exist_ok=True)

                for fr in sorted(int(x) for x in frames):
                    coords = row_to_coords(fr)
                    if coords is None:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print(f"[warn] {video_key}: cannot read frame {fr}")
                        continue
                    _draw_keypoints_bgr(frame, coords)
                    zoom = _crop_zoom(frame, coords, margin=120, scale=2.0)
                    if zoom is not None:
                        zoom_path = os.path.join(out_dir, f"frame_{fr:06d}_zoom.png")
                        cv2.imwrite(zoom_path, zoom)
                        per_video_saved.append(fr)
        finally:
            cap.release()

        saved[str(video_key)] = per_video_saved

    return saved

def run_keyframe_coord_collection():
    """Read keyframes_starts_only.json and export coords at each keyframe per video."""
    base_dir = os.path.dirname(__file__)
    starts_json = os.path.join(base_dir, "1-keyframes_starts_only.json")
    h5_dir = os.path.join(base_dir, "..", "videos")
    output = os.path.join(base_dir, "2-keyframe_coords.json")

    if not os.path.isfile(starts_json):
        raise FileNotFoundError(f"Not found: {starts_json}. Please generate starts_only first.")
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"Not found: {h5_dir}. Please ensure videos/.h5 are present.")

    data = collect_keyframe_coords(starts_json, h5_dir, output_path=output, how="union")
    print(f"Wrote coords for {len(data)} videos -> {output}")

def run_keyframe_image_export():
    """Export keyframe images with overlaid keypoints into videos/keyframe_image."""
    base_dir = os.path.dirname(__file__)
    starts_json = os.path.join(base_dir, "1-keyframes_starts_only.json")
    video_dir = os.path.join(base_dir, "..", "videos")
    h5_dir = video_dir  # h5 files live alongside videos

    if not os.path.isfile(starts_json):
        raise FileNotFoundError(f"Not found: {starts_json}. Please generate starts_only first.")
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Not found: {video_dir}. Please ensure videos/ are present.")

    saved = save_keyframe_images(starts_json, h5_dir=h5_dir, video_dir=video_dir)
    total_frames = sum(len(v) for v in saved.values())
    print(f"Wrote {total_frames} keyframe images across {len(saved)} videos -> {os.path.join(video_dir, 'keyframe_image')}")

def detect_walk_direction(
    video_key: str,
    coords_data: Optional[Dict[str, Any]] = None,
    coords_path: Optional[str] = None,
    anchor_part: str = "F_elbow",
) -> str:
    """
    Infer walking direction from keyframe coordinates.

    Rule: ΔX = X(last) - X(first) for an anchor body part.
    - ΔX > 0 → 'right'
    - ΔX < 0 → 'left'
    - Otherwise or insufficient data → 'unknown'

    Works with both union and by_leg structures. Attempts to use `anchor_part`
    (default 'F_elbow'). If missing, falls back to 'nose', 'M_back', then 'F_back'.
    """
    # Load coords data if not provided
    if coords_data is None:
        if coords_path is None:
            base_dir = os.path.dirname(__file__)
            coords_path = os.path.join(base_dir, "2-keyframe_coords.json")
        with open(coords_path, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

    payload = coords_data.get(video_key)
    if payload is None:
        return "unknown"

    def _gather_items(pay: Any) -> List[Dict[str, Any]]:
        if isinstance(pay, list):
            return [it for it in pay if isinstance(it, dict)]
        elif isinstance(pay, dict):
            items: List[Dict[str, Any]] = []
            for leg in ("FL", "FR", "BL", "BR"):
                seq = pay.get(leg, []) or []
                items.extend([it for it in seq if isinstance(it, dict)])
            return items
        return []

    items = _gather_items(payload)
    if not items:
        return "unknown"

    anchor_candidates = [anchor_part, "nose", "M_back", "F_back"]

    def _extract_series(part_key: str) -> List[Tuple[int, float]]:
        pairs: List[Tuple[int, float]] = []
        for it in items:
            fr = it.get("frame")
            coords = it.get("coords")
            if fr is None or not isinstance(coords, dict):
                continue
            base = coords.get(part_key)
            if not isinstance(base, dict):
                continue
            x = base.get("x")
            if x is None:
                continue
            try:
                pairs.append((int(fr), float(x)))
            except Exception:
                continue
        pairs.sort(key=lambda t: t[0])
        return pairs

    series: List[Tuple[int, float]] = []
    for pk in anchor_candidates:
        series = _extract_series(pk)
        if len(series) >= 2:
            break

    if len(series) < 2:
        return "unknown"

    dx = series[-1][1] - series[0][1]
    if dx > 0:
        return "right"
    elif dx < 0:
        return "left"
    else:
        return "unknown"

if __name__ == "__main__":
    # Run the keyframe coordinate extraction end-to-end in this file
    run_keyframe_coord_collection()
    try:
        run_keyframe_image_export()
    except ImportError as e:
        print(f"[warn] Skipped keyframe image export: {e}")
    except Exception as e:
        print(f"[warn] Keyframe image export failed: {e}")
