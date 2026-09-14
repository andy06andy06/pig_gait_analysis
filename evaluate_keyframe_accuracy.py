import os
import json
import numpy as np
import pandas as pd

def evaluate_accuracy(gt_path="keyframe_ground_truth.json", 
                      pred_path="1-keyframes_starts_only.json", 
                      fps=120.0,
                      match_window=15):
    """
    Compare manual ground truth keyframes against algorithm predictions.
    
    Parameters
    ----------
    gt_path : str
        Path to JSON file containing manually annotated ground truth frames.
    pred_path : str
        Path to 1-keyframes_starts_only.json.
    fps : float
        Video frame rate (default 120.0 fps for high-speed gait videos).
    match_window : int
        Maximum frame distance to pair a ground truth frame with a predicted frame.
    """
    if not os.path.exists(gt_path):
        print(f"[Error] Ground truth file '{gt_path}' not found!")
        print("Please create it or use 'keyframe_ground_truth_template.json' as a base.")
        return None, None

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    records = []
    unmatched_gt = 0
    unmatched_pred = 0

    limbs = ["FL", "FR", "BL", "BR"]

    for vid, gt_limbs in gt_data.items():
        if vid not in pred_data:
            print(f"[Warning] Video '{vid}' in GT not found in predictions. Skipping.")
            continue
        
        pred_limbs = pred_data[vid]

        for limb in limbs:
            gt_list = gt_limbs.get(limb, [])
            pred_list = pred_limbs.get(limb, [])

            # Greedy nearest neighbor matching within match_window
            matched_pred_indices = set()
            for gt_f in gt_list:
                best_diff = float("inf")
                best_pred_idx = -1

                for i, p_f in enumerate(pred_list):
                    if i in matched_pred_indices:
                        continue
                    diff = abs(p_f - gt_f)
                    if diff < best_diff:
                        best_diff = diff
                        best_pred_idx = i

                if best_diff <= match_window:
                    matched_pred_indices.add(best_pred_idx)
                    p_f = pred_list[best_pred_idx]
                    frame_err = p_f - gt_f
                    time_err_ms = (frame_err / fps) * 1000.0
                    records.append({
                        "video_id": vid,
                        "limb": limb,
                        "gt_frame": gt_f,
                        "pred_frame": p_f,
                        "frame_err": frame_err,
                        "abs_frame_err": abs(frame_err),
                        "time_err_ms": time_err_ms,
                        "abs_time_err_ms": abs(time_err_ms)
                    })
                else:
                    unmatched_gt += 1

            unmatched_pred += len(pred_list) - len(matched_pred_indices)

    if not records:
        print("[Notice] No matched keyframes found. Please check ground truth annotations.")
        return None, None

    df = pd.DataFrame(records)

    print("\n" + "=" * 65)
    print("        KEYFRAME POSITIONING ACCURACY EVALUATION REPORT")
    print(f"        (High-Speed Camera: {fps:.1f} FPS | 1 frame = {1000.0/fps:.2f} ms)")
    print("=" * 65)
    print(f"Total Evaluated Videos: {df['video_id'].nunique()}")
    print(f"Total Matched Keyframes: {len(df)}")
    if unmatched_gt > 0:
        print(f"Unmatched GT frames (missed): {unmatched_gt}")
    if unmatched_pred > 0:
        print(f"Unmatched Predicted frames (spurious): {unmatched_pred}")
    print("-" * 65)

    # Per limb & Overall summary
    summary_rows = []
    
    for limb in limbs + ["Overall"]:
        sub_df = df if limb == "Overall" else df[df["limb"] == limb]
        if len(sub_df) == 0:
            continue
        
        n = len(sub_df)
        bias_f = sub_df["frame_err"].mean()
        mae_f = sub_df["abs_frame_err"].mean()
        std_f = sub_df["abs_frame_err"].std()
        rmse_f = np.sqrt((sub_df["frame_err"] ** 2).mean())

        bias_ms = sub_df["time_err_ms"].mean()
        mae_ms = sub_df["abs_time_err_ms"].mean()
        rmse_ms = np.sqrt((sub_df["time_err_ms"] ** 2).mean())

        acc_1f = (sub_df["abs_frame_err"] <= 1).mean() * 100.0
        acc_2f = (sub_df["abs_frame_err"] <= 2).mean() * 100.0
        acc_3f = (sub_df["abs_frame_err"] <= 3).mean() * 100.0

        summary_rows.append({
            "Limb": limb,
            "N": n,
            "Bias (frames)": f"{bias_f:+.2f}",
            "Bias (ms)": f"{bias_ms:+.1f}",
            "RMSE (frames)": f"{rmse_f:.2f}",
            "RMSE (ms)": f"{rmse_ms:.1f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 65)

    # Save to Markdown table for thesis
    md_table = summary_df.to_markdown(index=False)
    out_md_path = "keyframe_accuracy_table.md"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Keyframe Positioning Accuracy Validation (Bias & RMSE)\n\n")
        f.write(f"- Video Sample Count: {df['video_id'].nunique()} videos\n")
        f.write(f"- Total Keyframes Evaluated: {len(df)}\n")
        f.write(f"- Video Frame Rate: {fps} FPS (~{1000.0/fps:.2f} ms/frame)\n\n")
        f.write(md_table + "\n")
    print(f"\n[Success] Markdown table saved to: {out_md_path}")

    try:
        from plot_keyframe_accuracy import generate_accuracy_plots
        generate_accuracy_plots(gt_path=gt_path, pred_path=pred_path, fps=fps, out_dir=os.path.dirname(gt_path) or ".")
    except Exception as e:
        print(f"[Warning] Could not generate plots automatically: {e}")

    return df, summary_df

if __name__ == "__main__":
    evaluate_accuracy()

