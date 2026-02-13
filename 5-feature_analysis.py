import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from textwrap import wrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
Number = Union[int, float]


def is_number_list(values: Any) -> bool:
    """Return True if values is a non-empty list of numbers (int/float)."""
    if not isinstance(values, list) or len(values) == 0:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def zscore_standardize(values: List[Number]) -> List[float]:
    """Z-score standardize a list of numeric values.

    If the standard deviation is 0 (all values equal), return zeros.
    """
    # Convert to float for consistent output
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return []
    mean = sum(xs) / n
    # population std (ddof=0). For small n this is fine for feature scaling.
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    if std == 0:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


def standardize_inplace(node: Any) -> None:
    """Recursively traverse a JSON-like structure and standardize any
    dict that has a 'values' key containing a list of numbers.

    Leaves the structure unchanged otherwise (e.g., 'frames', 'legs', 'unit').
    """
    if isinstance(node, dict):
        # Remove units everywhere
        if "unit" in node:
            del node["unit"]

        # Remove stride_time_seconds (and singular variant if present)
        for key in ["stride_time_seconds", "stride_time_second"]:
            if key in node:
                del node[key]

        # If this dict has a numeric 'values' list, standardize it
        if "values" in node and is_number_list(node["values"]):
            node["values"] = zscore_standardize(node["values"])

        # Recurse into child values after local edits
        for k in list(node.keys()):
            standardize_inplace(node[k])
    elif isinstance(node, list):
        for item in node:
            standardize_inplace(item)
    # Other types are left as-is


def iter_value_paths(node: Any, prefix: str = ""):
    """Yield (path, values) pairs for every numeric values list in a nested object."""
    if isinstance(node, dict):
        if "values" in node and is_number_list(node["values"]):
            yield prefix, [float(v) for v in node["values"]]
            return
        for key, value in node.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from iter_value_paths(value, next_prefix)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_value_paths(item, next_prefix)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix, [float(node)]


def summarize_values(values: List[float]) -> List[float]:
    """Return a fixed-length feature vector describing a numeric sequence."""
    n = len(values)
    if n == 0:
        return [0.0] * 7
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    min_v = min(values)
    max_v = max(values)
    sorted_vals = sorted(values)
    mid = n // 2
    if n % 2:
        median = sorted_vals[mid]
    else:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    mean_abs = sum(abs(v) for v in values) / n
    pos_ratio = sum(1 for v in values if v >= 0) / n
    return [mean, std, min_v, max_v, median, mean_abs, pos_ratio]


def euclidean_distance_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def aggregate_sequence(values: List[float]) -> float:
    """Aggregate一組標準化後的序列,回傳 RMS(均方根)值。"""
    if not values:
        return 0.0
    squared_sum = sum(float(v) ** 2 for v in values)
    return math.sqrt(squared_sum / len(values))


def format_feature_label(label: str, width: int = 18) -> str:
    if label == "cluster_label":
        return "cluster\nlabel"
    safe_label = label.replace("_", " ")
    if ":" in safe_label:
        prefix, suffix = safe_label.split(":", 1)
        prefix = prefix.replace(".", " \u2022 ")
        text = f"{prefix}: {suffix}"
    else:
        text = safe_label.replace(".", " \u2022 ")
    wrapped = wrap(text, width=width)
    return "\n".join(wrapped) if wrapped else text


def correlation_reorder_indices(
    corr_matrix: np.ndarray,
    keep_last: bool = False,
) -> List[int]:
    n = corr_matrix.shape[0]
    if n <= 2:
        return list(range(n))

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:  # noqa: BLE001
        return list(range(n))

    base_indices = list(range(n - 1)) if keep_last else list(range(n))
    if len(base_indices) <= 2:
        ordered = base_indices
    else:
        submatrix = corr_matrix[np.ix_(base_indices, base_indices)]
        distance = 1.0 - submatrix
        distance = np.clip(distance, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        order = leaves_list(linkage_matrix).tolist()
        ordered = [base_indices[idx] for idx in order]

    if keep_last:
        ordered.append(n - 1)
    return ordered


def build_feature_matrix(data: Dict[str, Any]) -> Tuple[List[str], List[List[float]], List[str]]:
    videos: List[str] = list(data.keys())
    path_to_values: Dict[str, Dict[str, List[float]]] = {}

    for video_id, video_payload in data.items():
        for path, values in iter_value_paths(video_payload):
            if "symmetry_ratio" not in path:
                continue
            path_to_values.setdefault(path, {})[video_id] = values

    if not path_to_values:
        return [], [], []

    stat_names = ["mean", "std", "min", "max", "median", "mean_abs", "pos_ratio"]
    feature_names: List[str] = []
    features_by_video: Dict[str, List[float]] = {vid: [] for vid in videos}

    for path in sorted(path_to_values):
        buckets = path_to_values[path]
        if not any(buckets.get(vid) for vid in videos):
            continue
        feature_names.extend(f"{path}:{stat}" for stat in stat_names)
        for vid in videos:
            values = buckets.get(vid)
            stats = summarize_values(values if values is not None else [])
            features_by_video[vid].extend(stats)

    feature_matrix = [features_by_video[vid] for vid in videos]
    return videos, feature_matrix, feature_names


def build_aggregated_feature_matrix(
    data: Dict[str, Any],
    aggregator=aggregate_sequence,
) -> Tuple[List[str], List[List[float]], List[str]]:
    videos: List[str] = list(data.keys())
    path_to_values: Dict[str, Dict[str, List[float]]] = {}

    for video_id, video_payload in data.items():
        for path, values in iter_value_paths(video_payload):
            if "symmetry_ratio" not in path:
                continue
            path_to_values.setdefault(path, {})[video_id] = values

    if not path_to_values:
        return [], [], []

    feature_names = sorted(path_to_values.keys())
    features_by_video: Dict[str, List[float]] = {vid: [] for vid in videos}

    for path in feature_names:
        buckets = path_to_values[path]
        for vid in videos:
            values = buckets.get(vid)
            agg_value = aggregator(values if values is not None else [])
            features_by_video[vid].append(float(agg_value))

    feature_matrix = [features_by_video[vid] for vid in videos]
    return videos, feature_matrix, feature_names


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if matrix.shape[0] == 0:
        raise ValueError("matrix contains no samples")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 2), dtype=float)

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top_components = eigvecs[:, order[: min(2, eigvecs.shape[1])]]
    projected = centered @ top_components

    if projected.shape[1] == 1:
        projected = np.hstack([projected, np.zeros((projected.shape[0], 1))])
    elif projected.shape[1] == 0:
        projected = np.zeros((projected.shape[0], 2))
    return projected


def compute_projection(
    matrix: List[List[float]],
    method: str = "tsne",
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] == 0:
        raise ValueError("Feature matrix empty; nothing to project.")

    requested = method.lower()
    if requested == "tsne" and array.shape[0] >= 2:
        try:
            from sklearn.manifold import TSNE  # type: ignore

            perplexity = max(5, min(30, array.shape[0] // 3))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                init="random",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(array)
            return coords, "tsne"
        except Exception as exc:  # noqa: BLE001
            print(f"t-SNE unavailable ({exc}); falling back to PCA.")

    coords = pca_2d(array)
    return coords, "pca"


def compute_correlation_matrix(matrix: List[List[float]]) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional")
    if array.shape[0] < 2:
        raise ValueError("Need at least two samples to compute correlations")

    if array.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    if array.shape[1] == 1:
        return np.ones((1, 1), dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(array, rowvar=False)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_analysis(
    output_json: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    standardized_path = here / "3-standardized_keyframe_features.json"

    if not standardized_path.exists():
        raise FileNotFoundError(
            "3-standardized_keyframe_features.json not found. Run main() to create it first."
        )

    with standardized_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    videos, feature_matrix, feature_names = build_feature_matrix(data)

    if not videos:
        raise ValueError("No data available for correlation analysis.")
    if not feature_names:
        raise ValueError("Feature extraction produced an empty feature set.")

    array = np.asarray(feature_matrix, dtype=float)
    stds = array.std(axis=0)
    const_features = [feature_names[i] for i, val in enumerate(stds) if val < 1e-6]

    corr_matrix = compute_correlation_matrix(feature_matrix)

    upper_indices = np.triu_indices(len(feature_names), k=1)
    pairs: List[Dict[str, Union[str, float]]] = []
    for i, j in zip(*upper_indices):
        pairs.append(
            {
                "feature_a": feature_names[i],
                "feature_b": feature_names[j],
                "correlation": float(corr_matrix[i, j]),
                "abs_correlation": float(abs(corr_matrix[i, j])),
            }
        )

    top_pairs = sorted(pairs, key=lambda entry: entry["abs_correlation"], reverse=True)[:top_k]

    result = {
        "feature_names": feature_names,
        "correlation_matrix": corr_matrix.tolist(),
        "top_abs_correlations": top_pairs,
        "constant_features": const_features,
    }

    json_path = Path(output_json) if output_json else here / "5-feature_correlation_matrix.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    size = max(8.0, 0.25 * len(feature_names))
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_names, fontsize=6)
    ax.set_title("Feature Pearson Correlation Matrix (Detail)", fontsize=12)
    ax.grid(False)
    fig.tight_layout()

    img_dir = here / "feature_analysis"
    img_dir.mkdir(exist_ok=True)
    png_path = Path(output_png) if output_png else img_dir / "feature_correlation_matrix.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Correlation matrix saved to {json_path} and heatmap saved to {png_path}.")
    if const_features:
        print(
            "Constant features (zero variance across videos) were assigned zero correlations:"
        )
        for name in const_features:
            print(f"  - {name}")

    print("Top correlated feature pairs (by absolute value):")
    for entry in top_pairs:
        print(
            f"  {entry['feature_a']} vs {entry['feature_b']} -> correlation {entry['correlation']:+.3f}"
        )

    return result


def correlation_analysis_integrated(
    kmeans_output: Optional[Dict[str, Any]] = None,
    output_json: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
    aggregator=aggregate_sequence,
    include_cluster: bool = True,
    top_k: int = 10,
) -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    standardized_path = here / "3-standardized_keyframe_features.json"

    if not standardized_path.exists():
        raise FileNotFoundError(
            "3-standardized_keyframe_features.json not found. Run main() to create it first."
        )

    with standardized_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    videos, feature_matrix, feature_names = build_aggregated_feature_matrix(
        data, aggregator=aggregator
    )

    if not videos:
        raise ValueError("No data available for correlation analysis.")
    if not feature_names:
        raise ValueError("Feature extraction produced an empty feature set.")

    assignments: Optional[Dict[str, int]] = None
    if include_cluster:
        if kmeans_output is None:
            clusters_path = here / "5-kmeans_clusters.json"
            if clusters_path.exists():
                with clusters_path.open("r", encoding="utf-8") as f:
                    kmeans_output = json.load(f)
            else:
                kmeans_output = k_means()

        assignments_map = kmeans_output.get("assignments") if kmeans_output else None
        if not assignments_map:
            raise ValueError("kmeans_output does not contain assignments for clustering.")
        assignments = {
            str(key): int(value) for key, value in assignments_map.items()
        }

    matrix_with_cluster: List[List[float]] = []
    for vid, row in zip(videos, feature_matrix):
        augmented_row = list(row)
        if assignments is not None:
            if vid not in assignments:
                raise ValueError(f"Missing cluster assignment for video {vid}.")
            augmented_row.append(float(assignments[vid]))
        matrix_with_cluster.append(augmented_row)

    augmented_feature_names = list(feature_names)
    if assignments is not None:
        augmented_feature_names.append("cluster_label")

    array = np.asarray(matrix_with_cluster, dtype=float)
    stds = array.std(axis=0)
    const_features = [
        augmented_feature_names[i] for i, val in enumerate(stds) if val < 1e-6
    ]

    corr_matrix = compute_correlation_matrix(matrix_with_cluster)

    order = correlation_reorder_indices(
        corr_matrix, keep_last=assignments is not None
    )
    corr_matrix = corr_matrix[np.ix_(order, order)]
    augmented_feature_names = [augmented_feature_names[idx] for idx in order]

    upper_indices = np.triu_indices(len(augmented_feature_names), k=1)
    pairs: List[Dict[str, Union[str, float]]] = []
    for i, j in zip(*upper_indices):
        pairs.append(
            {
                "feature_a": augmented_feature_names[i],
                "feature_b": augmented_feature_names[j],
                "correlation": float(corr_matrix[i, j]),
                "abs_correlation": float(abs(corr_matrix[i, j])),
            }
        )

    top_pairs = sorted(pairs, key=lambda entry: entry["abs_correlation"], reverse=True)[:top_k]

    result = {
        "feature_names": augmented_feature_names,
        "correlation_matrix": corr_matrix.tolist(),
        "top_abs_correlations": top_pairs,
        "constant_features": const_features,
        "aggregator": aggregator.__name__ if hasattr(aggregator, "__name__") else str(aggregator),
    }

    json_path = (
        Path(output_json)
        if output_json
        else here / "5-feature_correlation_matrix_integrated.json"
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    formatted_labels = [format_feature_label(name, width=18) for name in augmented_feature_names]

    size = max(9.0, 0.32 * len(augmented_feature_names))
    fig, ax = plt.subplots(figsize=(size*2, size*2))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(augmented_feature_names)))
    ax.set_yticks(range(len(augmented_feature_names)))
    ax.set_xticklabels(formatted_labels, fontsize=7)
    ax.set_yticklabels(formatted_labels, fontsize=7)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(90)
        # tick_label.set_verticalalignment("center")
        tick_label.set_horizontalalignment("center")
    ax.tick_params(axis="x", pad=6, labelsize=7)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xticks(np.arange(-0.5, len(augmented_feature_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(augmented_feature_names), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor("#f7f7f7")
    ax.set_title("Aggregated Feature Pearson Correlation Matrix")

    for i in range(len(augmented_feature_names)):
        for j in range(len(augmented_feature_names)):
            if i > j:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="none",
                    zorder=3,
                )
                ax.add_patch(rect)
                value = corr_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    zorder=4,
                )

    grid_color = "black"
    grid_width = 0.3
    for idx in range(len(augmented_feature_names) + 1):
        ax.axhline(idx - 0.5, color=grid_color, linewidth=grid_width, zorder=5)
        ax.axvline(idx - 0.5, color=grid_color, linewidth=grid_width, zorder=5)

    fig.tight_layout(rect=[0.12, 0.15, 0.98, 0.96])

    img_dir = here / "feature_analysis"
    img_dir.mkdir(exist_ok=True)
    png_path = (
        Path(output_png)
        if output_png
        else img_dir / "feature_correlation_matrix_integrated.png"
    )
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(
        f"Integrated correlation matrix saved to {json_path} and heatmap saved to {png_path}."
    )
    if const_features:
        print(
            "Constant features (zero variance across videos) were assigned zero correlations:"
        )
        for name in const_features:
            print(f"  - {name}")

    print("Top correlated feature pairs (by absolute value):")
    for entry in top_pairs:
        print(
            f"  {entry['feature_a']} vs {entry['feature_b']} -> correlation {entry['correlation']:+.3f}"
        )

    return result


def summarize_cluster_differences(
    kmeans_output: Dict[str, Any],
    top_n: int = 5,
) -> Dict[int, List[Dict[str, float]]]:
    feature_names = kmeans_output.get("feature_names")
    centroids = kmeans_output.get("centroids")

    if not feature_names or centroids is None:
        raise ValueError("kmeans_output must contain feature_names and centroids.")

    centroid_array = np.asarray(centroids, dtype=float)
    if centroid_array.ndim != 2:
        raise ValueError("centroids must be a 2D array-like object.")

    global_mean = centroid_array.mean(axis=0)
    summaries: Dict[int, List[Dict[str, float]]] = {}

    for cluster_idx, centroid in enumerate(centroid_array):
        deviations = centroid - global_mean
        ranked_indices = np.argsort(np.abs(deviations))[::-1][:top_n]
        cluster_summary: List[Dict[str, float]] = []
        for feature_idx in ranked_indices:
            cluster_summary.append(
                {
                    "feature": feature_names[feature_idx],
                    "centroid_value": float(centroid[feature_idx]),
                    "deviation": float(deviations[feature_idx]),
                }
            )
        summaries[cluster_idx] = cluster_summary

    return summaries


def k_means(
    k: int = 2,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    aggregator=aggregate_sequence,
):
    here = Path(__file__).resolve().parent
    standardized_path = here / "3-standardized_keyframe_features.json"

    if not standardized_path.exists():
        raise FileNotFoundError(
            "3-standardized_keyframe_features.json not found. Run main() to create it first."
        )

    with standardized_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    videos, feature_matrix, feature_names = build_aggregated_feature_matrix(
        data, aggregator=aggregator
    )

    if not videos:
        raise ValueError("No data available for clustering.")
    if not feature_names:
        raise ValueError("Feature extraction produced an empty feature set.")

    num_samples = len(feature_matrix)

    if num_samples == 0:
        raise ValueError("No samples available for clustering.")

    if k <= 0:
        raise ValueError("k must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    k = min(k, num_samples)

    random.seed(42)
    initial_indices = random.sample(range(num_samples), k)
    centroids = [feature_matrix[idx][:] for idx in initial_indices]
    assignments = [-1] * num_samples

    for _ in range(max_iter):
        new_assignments: List[int] = []
        centroid_shift = 0.0
        clusters: List[List[List[float]]] = [[] for _ in range(k)]

        for point in feature_matrix:
            distances = [euclidean_distance_sq(point, centroid) for centroid in centroids]
            cluster_idx = min(range(k), key=lambda idx: distances[idx])
            new_assignments.append(cluster_idx)
            clusters[cluster_idx].append(point)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        for idx in range(k):
            if not clusters[idx]:
                centroids[idx] = feature_matrix[random.randrange(num_samples)][:]
                continue
            old_centroid = centroids[idx]
            dim = len(old_centroid)
            sums = [0.0] * dim
            for point in clusters[idx]:
                for d in range(dim):
                    sums[d] += point[d]
            new_centroid = [s / len(clusters[idx]) for s in sums]
            centroid_shift = max(
                centroid_shift,
                math.sqrt(euclidean_distance_sq(old_centroid, new_centroid)),
            )
            centroids[idx] = new_centroid

        if centroid_shift <= tolerance:
            break

    inertia = sum(
        euclidean_distance_sq(feature_matrix[i], centroids[assignments[i]])
        for i in range(num_samples)
    )

    summaries = summarize_cluster_differences(
        {
            "feature_names": feature_names,
            "centroids": centroids,
        }
    )

    output = {
        "k": k,
        "inertia": inertia,
        "assignments": {vid: assignments[idx] for idx, vid in enumerate(videos)},
        "centroids": centroids,
        "feature_names": feature_names,
        "cluster_summaries": summaries,
        "aggregator": aggregator.__name__ if hasattr(aggregator, "__name__") else str(aggregator),
    }

    output_path = here / "5-kmeans_clusters.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    for cluster_idx in range(k):
        member_count = sum(1 for idx in assignments if idx == cluster_idx)
        print(f"Cluster {cluster_idx}: {member_count} videos")
        for entry in summaries.get(cluster_idx, [])[:3]:
            feature = entry["feature"]
            deviation = entry["deviation"]
            print(f"  ↳ {feature} (deviation {deviation:+.3f})")

    print(f"K-means complete. Inertia: {inertia:.3f}. Results written to {output_path}.")

    return output


def visualize_clusters(
    kmeans_output: Optional[Dict[str, Any]] = None,
    projection: str = "tsne",
    annotate: bool = True,
    figsize: Tuple[float, float] = (8.0, 6.0),
) -> Path:
    here = Path(__file__).resolve().parent
    standardized_path = here / "3-standardized_keyframe_features.json"

    if not standardized_path.exists():
        raise FileNotFoundError(
            "3-standardized_keyframe_features.json not found. Run main() to create it first."
        )

    with standardized_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    videos, feature_matrix, _ = build_feature_matrix(data)

    if not videos:
        raise ValueError("No data available for visualization.")

    if kmeans_output is None:
        kmeans_output = k_means()

    assignments_map = kmeans_output.get("assignments")
    if not assignments_map:
        raise ValueError("kmeans_output does not contain assignments.")

    missing = [vid for vid in videos if vid not in assignments_map]
    if missing:
        raise ValueError(f"Missing cluster assignments for videos: {missing}")

    assignments = [assignments_map[vid] for vid in videos]
    coords, method_used = compute_projection(feature_matrix, method=projection)
    coords = np.asarray(coords, dtype=float)

    if coords.shape[1] != 2:
        raise ValueError("Projection did not return 2D coordinates.")

    clusters_array = np.asarray(assignments)
    unique_clusters = sorted(set(assignments))

    vivid_palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#ffff33",
        "#1b9e77",
    ]

    pca_palette = [
        "#d7191c",
        "#2c7bb6",
        "#1a9641",
        "#fdae61",
        "#abdda4",
        "#f46d43",
        "#66c2a5",
        "#3288bd",
        "#b2182b",
    ]

    palette_source = pca_palette if method_used == "pca" else vivid_palette

    if len(unique_clusters) <= len(palette_source):
        cluster_colors = palette_source[: len(unique_clusters)]
    else:
        cluster_colors = [
            plt.cm.gist_ncar(val)
            for val in np.linspace(0, 1, len(unique_clusters))
        ]

    auto_annotate = annotate or method_used == "pca"

    fig, ax = plt.subplots(figsize=figsize)
    for idx, cluster in enumerate(unique_clusters):
        mask = clusters_array == cluster
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cluster_colors[idx],
            label=f"Cluster {cluster}",
            s=70,
            alpha=0.9,
            edgecolors="k",
            linewidth=0.4,
        )

    if auto_annotate:
        for i, vid in enumerate(videos):
            ax.text(
                coords[i, 0],
                coords[i, 1],
                str(vid),
                fontsize=7,
                alpha=0.8,
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "alpha": 0.6,
                    "ec": "none",
                },
            )

    ax.set_title(f"K-means clusters projected via {method_used.upper()}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    img_dir = here / "feature_analysis"
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / f"cluster_{method_used}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(
        f"Cluster projection saved to {output_path} using {method_used.upper()} projection."
    )

    return output_path


def plot_cluster_bars(
    kmeans_output: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
    figsize: Tuple[float, float] = (10.0, 6.0),
    aggregator=aggregate_sequence,
) -> Path:
    here = Path(__file__).resolve().parent
    if kmeans_output is None:
        clusters_path = here / "5-kmeans_clusters.json"
        if not clusters_path.exists():
            raise FileNotFoundError(
                "5-kmeans_clusters.json not found. Run k_means() or main() first."
            )
        with clusters_path.open("r", encoding="utf-8") as f:
            kmeans_output = json.load(f)

    assignments_map = kmeans_output.get("assignments") if kmeans_output else None
    if not assignments_map:
        raise ValueError("kmeans_output does not contain assignments.")

    standardized_path = here / "3-standardized_keyframe_features.json"
    if not standardized_path.exists():
        raise FileNotFoundError(
            "3-standardized_keyframe_features.json not found. Run main() to create it first."
        )
    with standardized_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    videos, feature_matrix, feature_names = build_aggregated_feature_matrix(
        data, aggregator=aggregator
    )

    if not videos or not feature_names:
        raise ValueError("Aggregated feature extraction produced no data to plot.")

    array = np.asarray(feature_matrix, dtype=float)
    global_mean = array.mean(axis=0)

    normalized_assignments = {str(key): int(val) for key, val in assignments_map.items()}
    cluster_sizes = Counter(normalized_assignments.values())
    cluster_indices = sorted(cluster_sizes.keys())
    if not cluster_indices:
        raise ValueError("No clusters found in assignments.")

    deviations_by_cluster: Dict[int, List[Tuple[str, float]]] = {}
    all_deviations: List[float] = []

    for cluster_idx in cluster_indices:
        selected_indices = [
            idx
            for idx, vid in enumerate(videos)
            if normalized_assignments.get(str(vid)) == cluster_idx
        ]
        if not selected_indices:
            continue
        cluster_mean = array[selected_indices].mean(axis=0)
        deviations = cluster_mean - global_mean
        ranked = np.argsort(np.abs(deviations))[::-1][:top_n]
        deviations_by_cluster[cluster_idx] = [
            (feature_names[i], float(deviations[i])) for i in ranked
        ]
        all_deviations.extend(float(deviations[i]) for i in ranked)

    if not deviations_by_cluster:
        raise ValueError("Unable to compute deviations for any cluster.")

    global_max = max((abs(dev) for dev in all_deviations), default=1.0)
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-global_max, global_max)

    num_clusters = len(cluster_indices)
    fig, axes = plt.subplots(
        num_clusters,
        1,
        figsize=(figsize[0], figsize[1] * num_clusters),
        squeeze=False,
    )

    for row, cluster_idx in enumerate(cluster_indices):
        ax = axes[row][0]
        entries = deviations_by_cluster.get(cluster_idx, [])
        labels = [format_feature_label(item[0], width=28) for item in entries]
        deviations = [item[1] for item in entries]
        y_positions = np.arange(len(labels))

        colors = cmap(norm(deviations)) if deviations else cmap(norm([0]))
        bars = ax.barh(
            y_positions,
            deviations,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            height=0.6,
        )

        max_range = max(global_max, max((abs(val) for val in deviations), default=1.0))
        max_range *= 1.15

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(-max_range, max_range)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_facecolor("#f5f5f5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.spines["bottom"].set_color("#999999")

        for bar, value in zip(bars, deviations):
            text_offset = 0.03 * max_range
            text_x = (
                bar.get_width() + text_offset
                if value >= 0
                else bar.get_width() - text_offset
            )
            ha = "left" if value >= 0 else "right"
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha=ha,
                fontsize=8,
                color="black",
            )

        count = cluster_sizes.get(cluster_idx, 0)
        ax.set_xlabel("Deviation from global aggregated mean")
        ax.set_title(
            f"Cluster {cluster_idx} (n={count}) top {top_n} drivers",
            fontsize=11,
            pad=10,
        )
        ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    axis_list = [axes[idx][0] for idx in range(num_clusters)]

    fig.subplots_adjust(left=0.3, right=0.8, top=0.96, bottom=0.06, hspace=0.6)

    cbar = fig.colorbar(
        sm,
        ax=axis_list,
        fraction=0.05,
        pad=0.02,
        label="Deviation",
    )
    cbar.ax.tick_params(labelsize=9)

    img_dir = here / "feature_analysis"
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / "cluster_feature_deviations.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Cluster feature bar charts saved to {output_path}.")

    return output_path


def main() -> None:
    here = Path(__file__).resolve().parent
    input_path = here / "3-keyframe_features.json"
    output_path = here / "3-standardized_keyframe_features.json"

    with input_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # standardize_inplace(data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    aggregator_fn = aggregate_sequence

    print("Standardization complete. Running k-means clustering...")
    kmeans_result = k_means(aggregator=aggregator_fn)
    print("Generating cluster visualization (t-SNE with PCA fallback)...")
    visualize_clusters(kmeans_result, projection="tsne")
    print("Generating PCA cluster visualization with labels...")
    visualize_clusters(kmeans_result, projection="pca", annotate=True)
    print("Plotting top feature deviations per cluster...")
    plot_cluster_bars(kmeans_result, aggregator=aggregator_fn)
    # print("Computing feature Pearson correlation matrix...")
    # correlation_analysis()
    print("Computing aggregated feature Pearson correlation matrix (with cluster label)...")
    correlation_analysis_integrated(kmeans_result)


if __name__ == "__main__":
    main()
