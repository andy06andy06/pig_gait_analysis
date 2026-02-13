import json
import os
from typing import Dict, List, Optional, Any, Tuple
import math

def _normalize_hoof_name(hoof: str, part: str = "hoof") -> str:
    """Return the coords base key used in keyframe_coords (e.g., 'FL_hoof')."""
    hoof = hoof.strip()
    if hoof.endswith(f"_{part}"):
        return hoof
    return f"{hoof}_{part}"

def _is_by_leg_structure(sample_value: Any) -> bool:
    """Detect whether keyframe_coords for a video is in by_leg format."""
    if isinstance(sample_value, dict):
        keys = set(sample_value.keys())
        return {"FL", "FR", "BL", "BR"}.issubset(keys)
    return False

def stride_time(
    keyframe_coords: Dict[str, Any],
    hoof: str = "FL",
    starts_only: Optional[Dict[str, Dict[str, List[int]]]] = None,
) -> Dict[str, List[int]]:
    """
    Compute frame gaps (stride time, in frames) between consecutive keyframes for the specified hoof.

    Inputs
    - keyframe_coords: Loaded JSON from code/keyframe_coords.json
      Two supported shapes per video_key:
        1) union: [ {"frame": int, "coords": {...}}, ... ]
        2) by_leg: {"FL": [...], "FR": [...], "BL": [...], "BR": [...]} where each list item matches union item shape.
    - hoof: One of 'FL', 'FR', 'BL', 'BR' (or 'FL_hoof', etc.)
    - starts_only: Optional loaded JSON from code/keyframes_starts_only.json. Used when keyframe_coords is union to disambiguate per-hoof frames.

    Returns
    - Dict mapping video_key -> list of stride_time values (frames, int) for that hoof.
    """
    target_hoof = hoof if len(hoof) == 2 else hoof[:2]
    results: Dict[str, List[int]] = {}

    for video_key, payload in keyframe_coords.items():
        frames: List[int] = []

        if _is_by_leg_structure(payload):
            items = payload.get(target_hoof, []) or []
            frames = sorted(int(it["frame"]) for it in items if isinstance(it, dict) and "frame" in it)
        else:
            # union structure: support raw list or wrapper dict with 'items'
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                items = payload.get("items", [])
            else:
                items = []
            if starts_only and video_key in starts_only and target_hoof in starts_only[video_key]:
                whitelisted = set(int(x) for x in (starts_only[video_key][target_hoof] or []))
                frames = sorted(int(it["frame"]) for it in items if isinstance(it, dict) and "frame" in it and int(it["frame"]) in whitelisted)
            else:
                # Fallback: use all frames in union (may mix legs)
                frames = sorted(int(it["frame"]) for it in items if isinstance(it, dict) and "frame" in it)

        gaps = [b - a for a, b in zip(frames, frames[1:])]
        results[video_key] = gaps
    return results

def stance_time(
    segments_data: Dict[str, Any],
    video_key: str,
    hoof: str = "FL",
) -> List[int]:
    """
    Compute stance time (in frames) for the specified hoof using keyframes_segments.json.
    Stance time is calculated using start and end frame indices.
    """
    target_hoof = hoof if len(hoof) == 2 else hoof[:2]
    
    if video_key not in segments_data:
        return []
    
    hoof_segments = segments_data[video_key].get(target_hoof, [])
    if not isinstance(hoof_segments, list):
        return []

    results = []
    for seg in hoof_segments:
        if isinstance(seg, dict) and "start" in seg and "end" in seg:
            try:
                # Calculate stance time using end - start
                val = int(seg["end"]) - int(seg["start"])
                results.append(val)
            except (ValueError, TypeError):
                pass
    return results

def hoof_release_angle(
    video_key: str,
    frame: int,
    direction: Optional[str] = None,
    coords_data: Optional[Dict[str, Any]] = None,
    coords_path: Optional[str] = None,
    limb: str = "front",
) -> Dict[str, Optional[float]]:
    """
    Compute hoof release angles for a given video/frame using keyframe_coords.json.

    - Direction and leg selection:
      - If `direction` is not provided, prefer the video's top-level `direction` field
        in keyframe_coords (new union-wrapper format). Fallback default is 'right'.
      - limb='front':
          - 'right' → FR_knee/FR_hoof with F_elbow
          - 'left'  → FL_knee/FL_hoof with F_elbow
        limb='hind':
          - 'right' → BR_knee/BR_hoof with B_elbow
          - 'left'  → BL_knee/BL_hoof with B_elbow

    - Angles (degrees):
      - alpha1: angle between vector (F_elbow → knee) and the ground horizontal.
      - alpha2: angle between vector (knee → hoof) and the ground horizontal.
      - Returns the acute angle in [0, 90], rounded to 2 decimals; returns None if
        any required point is missing or vector is degenerate.

    - Data structures supported for the per-video payload:
      1) union (list): [ {"frame": int, "coords": {...}}, ... ]
      2) union-wrapper (dict): {"direction": str, "items": [ ... union items ... ]}
      3) by_leg (dict): {"FL": [...], "FR": [...], "BL": [...], "BR": [...]} with union item shape.
    """
    if coords_data is None:
        if coords_path is None:
            base_dir = os.path.dirname(__file__)
            coords_path = os.path.join(base_dir, "2-keyframe_coords.json")
        with open(coords_path, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

    if video_key not in coords_data:
        return {"alpha1_deg": None, "alpha2_deg": None, "leg": ("FR" if str(direction).lower().startswith("r") else "FL")}

    payload = coords_data[video_key]

    # Determine direction preference: use provided direction if given; otherwise
    # prefer top-level 'direction' in the new wrapper structure; else default 'right'.
    dir_used: Optional[str] = None
    if isinstance(direction, str) and direction:
        dir_used = direction
    elif isinstance(payload, dict) and isinstance(payload.get("direction"), str):
        dir_used = payload.get("direction")
    if not isinstance(dir_used, str) or dir_used.lower() not in ("right", "left"):
        dir_used = "right"

    def _find_coords_in_payload(payload_any: Any, target_frame: int) -> Optional[Dict[str, Any]]:
        if isinstance(payload_any, list):
            for it in payload_any:
                if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                    return it.get("coords")
        elif isinstance(payload_any, dict):
            # union wrapper with 'items'
            if isinstance(payload_any.get("items"), list):
                for it in payload_any.get("items", []):
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
            # by_leg 結構：從所有腿的列表中搜尋該幀
            for leg_key in ("FL", "FR", "BL", "BR"):
                items = payload_any.get(leg_key, []) or []
                for it in items:
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
        return None

    coords = _find_coords_in_payload(payload, frame)
    is_right = str(dir_used).lower().startswith("r")
    if str(limb).lower().startswith("h"):  # hind
        leg = "BR" if is_right else "BL"
        elbow_key = "B_elbow"
    else:  # front (default)
        leg = "FR" if is_right else "FL"
        elbow_key = "F_elbow"
    if not isinstance(coords, dict):
        return {"alpha1_deg": None, "alpha2_deg": None, "leg": leg}

    def _get_point(key: str) -> Optional[Tuple[float, float]]:
        base = coords.get(key)
        if not isinstance(base, dict):
            return None
        x = base.get("x")
        y = base.get("y")
        if x is None or y is None:
            return None
        return float(x), float(y)

    def _acute_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
        x1, y1 = p1
        x2, y2 = p2
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        if dx == 0.0 and dy == 0.0:
            return None
        return round(math.degrees(math.atan2(abs(dy), abs(dx))), 2)

    F_elbow_pt = _get_point(elbow_key)
    knee_pt = _get_point(f"{leg}_knee")
    hoof_pt = _get_point(f"{leg}_hoof")

    if F_elbow_pt is None or knee_pt is None or hoof_pt is None:
        return {"alpha1_deg": None, "alpha2_deg": None, "leg": leg}

    alpha1 = _acute_angle_deg(F_elbow_pt, knee_pt)
    alpha2 = _acute_angle_deg(knee_pt, hoof_pt)
    return {"alpha1_deg": alpha1, "alpha2_deg": alpha2, "leg": leg}

def back_neck_angle(
    video_key: str,
    frame: int,
    direction: Optional[str] = None,
    coords_data: Optional[Dict[str, Any]] = None,
    coords_path: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute back/neck angles for a given video/frame using keyframe_coords.json.

    Points along the body axis: nose, F_back, M_back, B_back.
    - beta1: angle at F_back formed by (nose, F_back, M_back)
    - beta2: angle at M_back formed by (F_back, M_back, B_back)

    Returns {'beta1_deg': float|None, 'beta2_deg': float|None}.
    """
    if coords_data is None:
        if coords_path is None:
            base_dir = os.path.dirname(__file__)
            coords_path = os.path.join(base_dir, "2-keyframe_coords.json")
        with open(coords_path, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

    payload = coords_data.get(video_key)
    if payload is None:
        return {"beta1_deg": None, "beta2_deg": None}

    def _find_coords(payload_any: Any, target_frame: int) -> Optional[Dict[str, Any]]:
        if isinstance(payload_any, list):
            for it in payload_any:
                if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                    return it.get("coords")
        elif isinstance(payload_any, dict):
            if isinstance(payload_any.get("items"), list):
                for it in payload_any.get("items", []):
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
            for leg_key in ("FL", "FR", "BL", "BR"):
                items = payload_any.get(leg_key, []) or []
                for it in items:
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
        return None

    coords = _find_coords(payload, frame)
    if not isinstance(coords, dict):
        return {"beta1_deg": None, "beta2_deg": None}

    def _get_point(key: str) -> Optional[Tuple[float, float]]:
        base = coords.get(key)
        if not isinstance(base, dict):
            return None
        x = base.get("x")
        y = base.get("y")
        if x is None or y is None:
            return None
        return float(x), float(y)

    def _angle_at_vertex(a: Tuple[float, float], v: Tuple[float, float], b: Tuple[float, float]) -> Optional[float]:
        ax, ay = a
        vx, vy = v
        bx, by = b
        v1x, v1y = ax - vx, ay - vy
        v2x, v2y = bx - vx, by - vy
        n1 = (v1x**2 + v1y**2) ** 0.5
        n2 = (v2x**2 + v2y**2) ** 0.5
        if n1 == 0 or n2 == 0:
            return None
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = 1 if dot > 1 else (-1 if dot < -1 else dot)
        ang = math.degrees(math.acos(dot))
        return round(ang, 2)

    nose = _get_point("nose")
    Fb = _get_point("F_back")
    Mb = _get_point("M_back")
    Hb = _get_point("B_back")

    beta1 = _angle_at_vertex(nose, Fb, Mb) if (nose and Fb and Mb) else None
    beta2 = _angle_at_vertex(Fb, Mb, Hb) if (Fb and Mb and Hb) else None

    return {"beta1_deg": beta1, "beta2_deg": beta2}

def back_height_feature(
    video_key: str,
    frame: int,
    direction: Optional[str] = None,
    coords_data: Optional[Dict[str, Any]] = None,
    coords_path: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute back height feature H for a given video/frame.

    h1 = B_back.y, h2 = M_back.y, h3 = F_back.y; H = (h1 + h3)/2 - h2
    Returns {'H': float|None}.
    """
    if coords_data is None:
        if coords_path is None:
            base_dir = os.path.dirname(__file__)
            coords_path = os.path.join(base_dir, "2-keyframe_coords.json")
        with open(coords_path, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

    payload = coords_data.get(video_key)
    if payload is None:
        return {"H": None}

    def _find_coords(payload_any: Any, target_frame: int) -> Optional[Dict[str, Any]]:
        if isinstance(payload_any, list):
            for it in payload_any:
                if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                    return it.get("coords")
        elif isinstance(payload_any, dict):
            if isinstance(payload_any.get("items"), list):
                for it in payload_any.get("items", []):
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
            for leg_key in ("FL", "FR", "BL", "BR"):
                items = payload_any.get(leg_key, []) or []
                for it in items:
                    if isinstance(it, dict) and int(it.get("frame", -1)) == int(target_frame):
                        return it.get("coords")
        return None

    coords = _find_coords(payload, frame)
    if not isinstance(coords, dict):
        return {"H": None}

    def _get_y(key: str) -> Optional[float]:
        base = coords.get(key)
        if not isinstance(base, dict):
            return None
        y = base.get("y")
        if y is None:
            return None
        return float(y)

    h1 = _get_y("B_back")
    h2 = _get_y("M_back")
    h3 = _get_y("F_back")
    if h1 is None or h2 is None or h3 is None:
        return {"H": None}
    H = (h1 + h3) / 2.0 - h2
    return {"H": round(H, 2)}

def stride_length(
    keyframe_coords: Dict[str, Any],
    hoof: str = "FL",
    starts_only: Optional[Dict[str, Dict[str, List[int]]]] = None,
) -> Dict[str, List[float]]:
    """
    Compute horizontal stride lengths (delta X) between consecutive keyframes for the specified hoof.

    Inputs/shape semantics are the same as stride_time().

    Returns
    - Dict mapping video_key -> list of stride lengths (float) for that hoof.
    """
    target_hoof = hoof if len(hoof) == 2 else hoof[:2]
    coord_key = _normalize_hoof_name(target_hoof, part="hoof")
    results: Dict[str, List[float]] = {}

    for video_key, payload in keyframe_coords.items():
        # Build per-video ordered frames and X series for this hoof
        def extract_series(items_list):
            pairs = []  # (frame, x)
            for it in items_list:
                if not isinstance(it, dict) or "frame" not in it or "coords" not in it:
                    continue
                fr = int(it["frame"])
                coords = it["coords"] or {}
                base = coords.get(coord_key)
                if isinstance(base, dict):
                    x = base.get("x")
                    if x is None:
                        continue
                    pairs.append((fr, float(x)))
            pairs.sort(key=lambda t: t[0])
            return pairs

        if _is_by_leg_structure(payload):
            items = payload.get(target_hoof, []) or []
            series = extract_series(items)
        else:
            # union structure: support raw list or wrapper dict with 'items'
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                items = payload.get("items", [])
            else:
                items = []
            if starts_only and video_key in starts_only and target_hoof in starts_only[video_key]:
                whitelist = set(int(x) for x in (starts_only[video_key][target_hoof] or []))
                items = [it for it in items if isinstance(it, dict) and int(it.get("frame", -1)) in whitelist]
            # else: use all union items as fallback
            series = extract_series(items)

        # Compute horizontal deltas between consecutive keyframes
        strides: List[float] = []
        for (_, x1), (_, x2) in zip(series, series[1:]):
            strides.append(abs(x2 - x1))
        results[video_key] = strides

    return results

def compute_symmetry(per_hoof_data: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute symmetry ratios for Stride Time and Stride Length.
    Mirroring the structure of 'symmetry_table' in the pressure mat JSON,
    excluding velocity and max force.
    """
    metrics = ["stride_time", "stride_length", "stance_time"]
    sections = [
        "Front / Hind",
        "Left / Right",
        "Left Front / Right Front",
        "Left Hind / Right Hind"
    ]
    
    def get_mean(values: List[Any]) -> Optional[float]:
        if not values:
            return None
        valid = [v for v in values if isinstance(v, (int, float))]
        if not valid:
            return None
        return sum(valid) / len(valid)

    hooves = ["FL", "FR", "BL", "BR"]
    means = {h: {} for h in hooves}
    for h in hooves:
        hoof_data = per_hoof_data.get(h, {})
        for m in metrics:
            means[h][m] = get_mean(hoof_data.get(m, []))

    results = {}
    for section in sections:
        results[section] = {}
        for m in metrics:
            val_numerator = None
            val_denominator = None
            
            if section == "Front / Hind":
                fl, fr = means["FL"].get(m), means["FR"].get(m)
                bl, br = means["BL"].get(m), means["BR"].get(m)
                nums = [x for x in (fl, fr) if x is not None]
                dens = [x for x in (bl, br) if x is not None]
                if nums: val_numerator = sum(nums) / len(nums)
                if dens: val_denominator = sum(dens) / len(dens)
                
            elif section == "Left / Right":
                fl, bl = means["FL"].get(m), means["BL"].get(m)
                fr, br = means["FR"].get(m), means["BR"].get(m)
                nums = [x for x in (fl, bl) if x is not None]
                dens = [x for x in (fr, br) if x is not None]
                if nums: val_numerator = sum(nums) / len(nums)
                if dens: val_denominator = sum(dens) / len(dens)

            elif section == "Left Front / Right Front":
                val_numerator = means["FL"].get(m)
                val_denominator = means["FR"].get(m)

            elif section == "Left Hind / Right Hind":
                val_numerator = means["BL"].get(m)
                val_denominator = means["BR"].get(m)

            if val_numerator is not None and val_denominator is not None and val_denominator != 0:
                results[section][m] = round(val_numerator / val_denominator, 2)
            else:
                results[section][m] = None
    return results

if __name__ == "__main__":
    # Load keyframe_coords.json and optional keyframes_starts_only.json
    base_dir = os.path.dirname(__file__)
    coords_path = os.path.join(base_dir, "2-keyframe_coords.json")
    starts_only_path = os.path.join(base_dir, "1-keyframes_starts_only.json")
    segments_path = os.path.join(base_dir, "1-keyframes_segments.json")
    output_path = os.path.join(base_dir, "3-keyframe_feature.json")

    if not os.path.isfile(coords_path):
        raise FileNotFoundError(f"Not found: {coords_path}")

    with open(coords_path, "r", encoding="utf-8") as f:
        keyframe_coords = json.load(f)

    starts_only = None
    if os.path.isfile(starts_only_path):
        with open(starts_only_path, "r", encoding="utf-8") as f:
            starts_only = json.load(f)

    segments_data = {}
    if os.path.isfile(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)

    # Compute gait features for all four hooves per video
    hooves = ("FL", "FR", "BL", "BR")
    features: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}

    # Precompute per-hoof data
    stride_time_all = {h: stride_time(keyframe_coords, hoof=h, starts_only=starts_only) for h in hooves}
    stride_length_all = {h: stride_length(keyframe_coords, hoof=h, starts_only=starts_only) for h in hooves}
    stance_time_all = {h: {} for h in hooves}
    for h in hooves:
        # Compute stance time for all videos
        for vk in keyframe_coords.keys():
            stance_time_all[h][vk] = stance_time(segments_data, video_key=vk, hoof=h)

    # Merge into per-video structure
    video_keys = set(keyframe_coords.keys())
    for vk in sorted(video_keys, key=str):
        features[vk] = {}
        for h in hooves:
            features[vk][h] = {
                "stride_time": stride_time_all.get(h, {}).get(vk, []),
                "stride_length": stride_length_all.get(h, {}).get(vk, []),
                "stance_time": stance_time_all.get(h, {}).get(vk, []),
            }

        # Compute hoof release angles using only the corresponding front-leg keyframes
        payload = keyframe_coords.get(vk)
        # Determine direction for this video (top-level if available)
        dir_for_vk = None
        if isinstance(payload, dict) and isinstance(payload.get("direction"), str):
            dir_for_vk = payload.get("direction")
        if not isinstance(dir_for_vk, str) or dir_for_vk.lower() not in ("right", "left"):
            dir_for_vk = "right"
        front_leg = "FR" if dir_for_vk.lower() == "right" else "FL"

        frames_for_vk: List[int] = []
        # Prefer starts_only when available
        if isinstance(starts_only, dict) and vk in starts_only and isinstance(starts_only[vk], dict):
            leg_frames = starts_only[vk].get(front_leg)
            if isinstance(leg_frames, list) and leg_frames:
                frames_for_vk = [int(x) for x in leg_frames]
        # Else, if payload is by_leg, use that leg's frames
        if not frames_for_vk and isinstance(payload, dict) and _is_by_leg_structure(payload):
            items = payload.get(front_leg, []) or []
            frames_for_vk = [int(it.get("frame", -1)) for it in items if isinstance(it, dict) and "frame" in it]
        # Else, as a final fallback, use union items' frames (may include other legs)
        if not frames_for_vk:
            if isinstance(payload, list):
                frames_for_vk = [int(it.get("frame", -1)) for it in payload if isinstance(it, dict) and "frame" in it]
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                frames_for_vk = [int(it.get("frame", -1)) for it in payload.get("items", []) if isinstance(it, dict) and "frame" in it]
        frames_for_vk = sorted({fr for fr in frames_for_vk if isinstance(fr, int) and fr >= 0})

        alpha1_vals: List[float] = []
        alpha2_vals: List[float] = []
        used_leg: Optional[str] = None
        for fr in frames_for_vk:
            ar = hoof_release_angle(vk, fr, direction=None, coords_data=keyframe_coords)
            if used_leg is None and isinstance(ar.get("leg"), str):
                used_leg = ar.get("leg")
            a1 = ar.get("alpha1_deg")
            a2 = ar.get("alpha2_deg")
            if isinstance(a1, (int, float)):
                alpha1_vals.append(float(a1))
            if isinstance(a2, (int, float)):
                alpha2_vals.append(float(a2))
        features[vk]["front_hoof_release_angle"] = {
            "leg": used_leg,
            "alpha1": alpha1_vals,
            "alpha2": alpha2_vals,
        }

        # Compute hind hoof release angles using only the corresponding hind-leg keyframes
        alpha1_vals_h: List[float] = []
        alpha2_vals_h: List[float] = []
        used_leg_h: Optional[str] = None
        # Determine hind leg (BR for right, BL for left)
        hind_leg = "BR" if dir_for_vk.lower() == "right" else "BL"

        frames_for_vk_h: List[int] = []
        if isinstance(starts_only, dict) and vk in starts_only and isinstance(starts_only[vk], dict):
            leg_frames_h = starts_only[vk].get(hind_leg)
            if isinstance(leg_frames_h, list) and leg_frames_h:
                frames_for_vk_h = [int(x) for x in leg_frames_h]
        if not frames_for_vk_h and isinstance(payload, dict) and _is_by_leg_structure(payload):
            items_h = payload.get(hind_leg, []) or []
            frames_for_vk_h = [int(it.get("frame", -1)) for it in items_h if isinstance(it, dict) and "frame" in it]
        if not frames_for_vk_h:
            if isinstance(payload, list):
                frames_for_vk_h = [int(it.get("frame", -1)) for it in payload if isinstance(it, dict) and "frame" in it]
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                frames_for_vk_h = [int(it.get("frame", -1)) for it in payload.get("items", []) if isinstance(it, dict) and "frame" in it]
        frames_for_vk_h = sorted({fr for fr in frames_for_vk_h if isinstance(fr, int) and fr >= 0})

        for fr in frames_for_vk_h:
            ar_h = hoof_release_angle(vk, fr, direction=dir_for_vk, coords_data=keyframe_coords, limb="hind")
            if used_leg_h is None and isinstance(ar_h.get("leg"), str):
                used_leg_h = ar_h.get("leg")
            a1h = ar_h.get("alpha1_deg")
            a2h = ar_h.get("alpha2_deg")
            if isinstance(a1h, (int, float)):
                alpha1_vals_h.append(float(a1h))
            if isinstance(a2h, (int, float)):
                alpha2_vals_h.append(float(a2h))
        features[vk]["hind_hoof_release_angle"] = {
            "leg": used_leg_h,
            "alpha1": alpha1_vals_h,
            "alpha2": alpha2_vals_h,
        }

        # Compute back/neck angles using union of keyframes across all legs
        # Build (frame, leg) pairs to retain provenance
        frames_union_pairs: List[Tuple[int, Optional[str]]] = []
        if isinstance(starts_only, dict) and vk in starts_only and isinstance(starts_only[vk], dict):
            for leg_key in ("FL", "FR", "BL", "BR"):
                seq = starts_only[vk].get(leg_key) or []
                if isinstance(seq, list):
                    for x in seq:
                        try:
                            frames_union_pairs.append((int(x), leg_key))
                        except Exception:
                            pass
            # Deduplicate per (frame, leg) and sort by frame
            seen = set()
            dedup: List[Tuple[int, Optional[str]]] = []
            for fr, lg in frames_union_pairs:
                key = (fr, lg)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((fr, lg))
            frames_union_pairs = sorted(dedup, key=lambda t: t[0])
        else:
            # Fallback from keyframe_coords payload
            if isinstance(payload, list):
                for it in payload:
                    if isinstance(it, dict) and "frame" in it:
                        try:
                            frames_union_pairs.append((int(it.get("frame", -1)), None))
                        except Exception:
                            pass
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                for it in payload.get("items", []):
                    if isinstance(it, dict) and "frame" in it:
                        try:
                            frames_union_pairs.append((int(it.get("frame", -1)), None))
                        except Exception:
                            pass
            elif isinstance(payload, dict):
                for leg_key in ("FL", "FR", "BL", "BR"):
                    seq = payload.get(leg_key, []) or []
                    for it in seq:
                        if isinstance(it, dict) and "frame" in it:
                            try:
                                frames_union_pairs.append((int(it.get("frame", -1)), leg_key))
                            except Exception:
                                pass
                # Deduplicate per (frame, leg) and sort
                seen = set()
                dedup: List[Tuple[int, Optional[str]]] = []
                for fr, lg in frames_union_pairs:
                    key = (fr, lg)
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append((fr, lg))
                frames_union_pairs = sorted(dedup, key=lambda t: t[0])

        beta1_vals: List[float] = []
        beta1_frames: List[int] = []
        beta1_legs: List[Optional[str]] = []
        beta2_vals: List[float] = []
        beta2_frames: List[int] = []
        beta2_legs: List[Optional[str]] = []
        for fr, leg_src in frames_union_pairs:
            # Only keep frames with explicit leg provenance
            if leg_src is None:
                continue
            br = back_neck_angle(vk, fr, direction=dir_for_vk, coords_data=keyframe_coords)
            b1 = br.get("beta1_deg")
            b2 = br.get("beta2_deg")
            if isinstance(b1, (int, float)):
                beta1_vals.append(float(b1))
                beta1_frames.append(int(fr))
                beta1_legs.append(leg_src)
            if isinstance(b2, (int, float)):
                beta2_vals.append(float(b2))
                beta2_frames.append(int(fr))
                beta2_legs.append(leg_src)

        features[vk]["back_neck_angle"] = {
            "beta1": beta1_vals,
            "beta1_frames": beta1_frames,
            "beta1_legs": beta1_legs,
            "beta2": beta2_vals,
            "beta2_frames": beta2_frames,
            "beta2_legs": beta2_legs,
        }

        # Compute back height feature across the same per-leg union frames
        H_vals: List[float] = []
        H_frames: List[int] = []
        H_legs: List[Optional[str]] = []
        for fr, leg_src in frames_union_pairs:
            if leg_src is None:
                continue
            hv = back_height_feature(vk, fr, direction=dir_for_vk, coords_data=keyframe_coords).get("H")
            if isinstance(hv, (int, float)):
                H_vals.append(float(hv))
                H_frames.append(int(fr))
                H_legs.append(leg_src)
        features[vk]["back_height_feature"] = {
            "H": H_vals,
            "H_frames": H_frames,
            "H_legs": H_legs,
        }

    # Attach units for readability and add seconds using 120 FPS
    fps = 120.0
    features_with_units: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for vk in sorted(video_keys, key=str):
        features_with_units[vk] = {}
        # Per-leg metrics
        for h in hooves:
            metrics = features.get(vk, {}).get(h, {}) or {}
            
            st_frames = metrics.get("stride_time", []) or []
            st_secs = [round((fr or 0) / fps, 2) for fr in st_frames]
            
            sl_vals_raw = metrics.get("stride_length", []) or []
            sl_vals = [round(v, 2) for v in sl_vals_raw]

            stance_frames = metrics.get("stance_time", []) or []
            stance_secs = [round((fr or 0) / fps, 2) for fr in stance_frames]
            
            features_with_units[vk][h] = {
                "stride_time": {"unit": "s", "values": st_secs},
                "stride_length": {"unit": "pixels", "values": sl_vals},
                "stance_time": {"unit": "s", "values": stance_secs},
            }

        # Hoof release angle metrics (front and hind)
        angles_front = features.get(vk, {}).get("front_hoof_release_angle")
        if isinstance(angles_front, dict):
            a1 = [round(v, 2) for v in (angles_front.get("alpha1") or [])]
            a2 = [round(v, 2) for v in (angles_front.get("alpha2") or [])]
            features_with_units[vk]["front_hoof_release_angle"] = {
                "leg": angles_front.get("leg"),
                "alpha1": {"unit": "deg", "values": a1},
                "alpha2": {"unit": "deg", "values": a2},
            }

        angles_hind = features.get(vk, {}).get("hind_hoof_release_angle")
        if isinstance(angles_hind, dict):
            a1h = [round(v, 2) for v in (angles_hind.get("alpha1") or [])]
            a2h = [round(v, 2) for v in (angles_hind.get("alpha2") or [])]
            features_with_units[vk]["hind_hoof_release_angle"] = {
                "leg": angles_hind.get("leg"),
                "alpha1": {"unit": "deg", "values": a1h},
                "alpha2": {"unit": "deg", "values": a2h},
            }

        # Back/neck angle metrics
        bna = features.get(vk, {}).get("back_neck_angle")
        if isinstance(bna, dict):
            b1 = [round(v, 2) for v in (bna.get("beta1") or [])]
            b2 = [round(v, 2) for v in (bna.get("beta2") or [])]
            features_with_units[vk]["back_neck_angle"] = {
                "beta1": {"unit": "deg", "values": b1, "frames": bna.get("beta1_frames", []), "legs": bna.get("beta1_legs", [])},
                "beta2": {"unit": "deg", "values": b2, "frames": bna.get("beta2_frames", []), "legs": bna.get("beta2_legs", [])},
            }

        # Back height feature metrics
        bhf = features.get(vk, {}).get("back_height_feature")
        if isinstance(bhf, dict):
            hv = [round(v, 2) for v in (bhf.get("H") or [])]
        features_with_units[vk]["back_height_feature"] = {
            "H": {"unit": "pixels", "values": hv, "frames": bhf.get("H_frames", []), "legs": bhf.get("H_legs", [])}
        }

        # Compute symmetry ratios
        features_with_units[vk]["symmetry_ratio"] = compute_symmetry(features.get(vk, {}))

    # Write output JSON (readable)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(features_with_units, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote gait features for {len(features)} videos -> {output_path}")
