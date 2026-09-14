import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_accuracy_plots(gt_path="keyframe_ground_truth.json",
                            pred_path="1-keyframes_starts_only.json",
                            fps=120.0,
                            out_dir="."):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    records = []
    limbs = ["FL", "FR", "BL", "BR"]

    for vid, gt_limbs in gt_data.items():
        if vid not in pred_data:
            continue
        pred_limbs = pred_data[vid]
        for limb in limbs:
            gt_list = gt_limbs.get(limb, [])
            pred_list = pred_limbs.get(limb, [])
            matched_pred_indices = set()
            for gt_f in gt_list:
                best_diff = float("inf")
                best_idx = -1
                for i, p_f in enumerate(pred_list):
                    if i in matched_pred_indices:
                        continue
                    diff = abs(p_f - gt_f)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
                if best_diff <= 15:
                    matched_pred_indices.add(best_idx)
                    p_f = pred_list[best_idx]
                    frame_err = p_f - gt_f
                    time_err_ms = (frame_err / fps) * 1000.0
                    records.append({
                        "video_id": vid,
                        "limb": limb,
                        "frame_err": frame_err,
                        "time_err_ms": time_err_ms
                    })

    df = pd.DataFrame(records)

    # Compute stats
    limb_order = ["FL", "FR", "BL", "BR", "Overall"]
    stats_list = []
    for limb in limb_order:
        sub = df if limb == "Overall" else df[df["limb"] == limb]
        bias_f = sub["frame_err"].mean()
        rmse_f = np.sqrt((sub["frame_err"]**2).mean())
        bias_ms = sub["time_err_ms"].mean()
        rmse_ms = np.sqrt((sub["time_err_ms"]**2).mean())
        stats_list.append({
            "limb": limb,
            "bias_f": bias_f,
            "rmse_f": rmse_f,
            "bias_ms": bias_ms,
            "rmse_ms": rmse_ms,
            "n": len(sub)
        })
    stats_df = pd.DataFrame(stats_list)

    # Matplotlib styling for publication quality
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.1

    # Single-figure canvas with balanced dimensions
    fig, ax1 = plt.subplots(figsize=(8.2, 5.2), dpi=300)

    x = np.arange(len(limb_order))
    width = 0.32

    bias_vals = stats_df["bias_ms"].values
    rmse_vals = stats_df["rmse_ms"].values

    # Distinct, publication-grade color scheme
    rects1 = ax1.bar(x - width/2, bias_vals, width, label="Bias", 
                     color="#3A86FF", alpha=0.88, edgecolor="#1B4D9B", linewidth=1.2, zorder=3)
    rects2 = ax1.bar(x + width/2, rmse_vals, width, label="RMSE", 
                     color="#E63946", alpha=0.88, edgecolor="#9E1B25", linewidth=1.2, zorder=3)

    # Solid, prominent zero baseline
    ax1.axhline(0, color="#222222", linestyle="-", linewidth=1.1, zorder=4)

    # Gridlines (underneath bars)
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, color="#E0E0E0", alpha=0.8, zorder=1)

    # Y limits to give comfortable breathing room
    y_min_ms = -20.0
    y_max_ms = 35.0
    ax1.set_ylim(y_min_ms, y_max_ms)

    # X and Y axis formatting (regular weight, no bold)
    ax1.set_ylabel("Temporal Error (ms)", fontsize=13, labelpad=10)
    ax1.set_xlabel("Leg", fontsize=13.5, labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(limb_order, fontsize=13)
    ax1.tick_params(axis="x", which="major", pad=8)
    ax1.tick_params(axis="y", which="major", labelsize=11.5, pad=6)

    # Annotate values above/below each bar with careful offset
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:+.1f} ms",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, -7), textcoords="offset points",
                     ha="center", va="top",
                     fontsize=10, fontweight="bold", color="#10366D")

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} ms",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#6E1118")

    # Enlarged Legend placed in upper-left
    ax1.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#C8C8C8",
               framealpha=0.96, fontsize=12, borderpad=0.9, labelspacing=0.6,
               handlelength=1.4, handleheight=1.0)

    # Ensure all labels fit perfectly without clipping
    plt.tight_layout(pad=1.8)

    out_png = os.path.join(out_dir, "Figure_Keyframe_Accuracy_Bias_RMSE.png")
    out_pdf = os.path.join(out_dir, "Figure_Keyframe_Accuracy_Bias_RMSE.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"[Success] Saved clean single-panel figure:\n  -> {out_png}\n  -> {out_pdf}")

if __name__ == "__main__":
    generate_accuracy_plots()
