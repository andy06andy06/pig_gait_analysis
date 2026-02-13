"""
用 DeepLabCut 匯出的 *keypoint-results.csv 繪製每個關鍵點的 Train/Test 誤差圖
同時輸出：
1. 不含 p-cutoff
2. 含 p-cutoff
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


csv_path = "../evaluation-results/iteration-0/pig_gait_v1Feb26-trainset95shuffle9/DLC_effnet_b0_pig_gait_v1Feb26shuffle9_100000-keypoint-results.csv"
out_base = "keypoint_errors"

df = pd.read_csv(csv_path)

# Bodyparts 名稱（跳過第一欄 Error Type）
bodyparts = df.columns[1:].tolist()

# 投影片指定的順序與標籤
slide_bodyparts = [
    "BR_hoof", "BR_knee", "BL_hoof", "BL_knee",
    "FL_hoof", "FL_knee", "FR_hoof", "FR_knee",
    "B_elbow", "F_elbow", "nose", "F_back",
    "M_back", "B_back", "B_improvement", "F_improvement",
]
slide_xtick_labels = [str(i) for i in range(1, 17)]

modes = [
    ("Train error (px)", "Test error (px)",            "_no_pcutoff"),
    ("Train error (px) with p-cutoff", "Test error (px) with p-cutoff", "_with_pcutoff"),
]

def get_row_values(df, error_type, bodyparts):
    row = df[df["Error Type"] == error_type]
    if row.empty:
        raise ValueError(f"找不到 {error_type}，CSV 內的 Error Type 有：{df['Error Type'].tolist()}")
    return row[bodyparts].iloc[0].astype(float).to_numpy()

def reorder_by_slide(vals, bodyparts, desired_order):
    mapping = {bp: v for bp, v in zip(bodyparts, vals)}
    missing = [bp for bp in desired_order if bp not in mapping]
    if missing:
        raise ValueError(f"以下 bodyparts 缺少數值: {missing}")
    return np.array([mapping[bp] for bp in desired_order])

def plot_grouped(train_vals, test_vals, bodyparts, title, out_path, xtick_labels=None, xtick_rotation=45, xtick_ha="right", legend_fontsize=12):
    x = np.arange(len(bodyparts))
    width = 0.35  # Width of the bars
    labels = xtick_labels if xtick_labels is not None else bodyparts
    
    # Use Paired colormap (Blue and Red) for a classic high-contrast look
    cmap = plt.get_cmap("Paired")
    color_train = cmap(1)  # Blue-ish
    color_test = cmap(5)   # Red-ish

    plt.figure(figsize=(12,6))
    bars_train = plt.bar(x - width/2, train_vals, width, label="Train error", color=color_train)
    bars_test  = plt.bar(x + width/2, test_vals, width, label="Test error", color=color_test)

    plt.xticks(x, labels, rotation=xtick_rotation, ha=xtick_ha)
    plt.xlabel("Keypoints")
    plt.ylabel("Error (px)")
    plt.title(title)
    plt.legend(fontsize=legend_fontsize)

    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            if h > 0:
                plt.text(b.get_x() + b.get_width()/2, h/2, f"{h:.2f}", 
                         ha="center", va="center", fontsize=10, 
                         rotation='vertical', color='white')

    add_labels(bars_train)
    add_labels(bars_test)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


for train_key, test_key, suffix in modes:
    train_vals = get_row_values(df, train_key, bodyparts)
    test_vals  = get_row_values(df, test_key, bodyparts)

    title = f"Per-keypoint Train/Test Error (px){' — with p-cutoff' if 'with p-cutoff' in train_key else ''}"
    out_path = f"../evaluation-results/iteration-0/pig_gait_v1Feb26-trainset95shuffle9/{out_base}{suffix}.png"
    plot_grouped(train_vals, test_vals, bodyparts, title, out_path)
    print(f"已輸出圖檔: {out_path}")

    # 額外輸出投影片用：按照指定順序重排並換上帶編號的新標籤
    train_slide = reorder_by_slide(train_vals, bodyparts, slide_bodyparts)
    test_slide  = reorder_by_slide(test_vals, bodyparts, slide_bodyparts)
    slide_title = f"Per-keypoint Train/Test Error (px) — slide order{' — with p-cutoff' if 'with p-cutoff' in train_key else ''}"
    slide_out_path = f"../evaluation-results/iteration-0/pig_gait_v1Feb26-trainset95shuffle9/{out_base}{suffix}_slide_ticks.png"
    plot_grouped(
        train_slide,
        test_slide,
        slide_bodyparts,
        slide_title,
        slide_out_path,
        xtick_labels=slide_xtick_labels,
        xtick_rotation=0,
        xtick_ha="center",
        legend_fontsize=14,
    )
    print(f"已輸出圖檔 (投影片 x tick): {slide_out_path}")
