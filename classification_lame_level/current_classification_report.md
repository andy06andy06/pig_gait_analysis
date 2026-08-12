# 豬隻分類現行標準與全量 Dataset 報告

本報告記錄了專案目前採用的豬隻跛行分類標準、權重計算公式、劃分門檻，以及依據此標準所分類產出的 **39 筆全量 Dataset 詳細資料** 與相應的 **SVM 機器學習模型驗證成果**。

---

## 一、 現行豬隻分類標準與計算公式 (Classification Standards & Formulas)

目前專案採用地面壓力墊 (Pressure Mat) 所測量之多維度運動學與力學數據，進行肢體左右不對稱性 (Asymmetry Score) 計算與標籤劃分。

### 1. 指標選用與權重分配 (Metric Weights)
為避免單一踩踏衝擊峰值帶來之誤差，現行標準融合了 5 大壓力墊指標：
* **最大峰值壓力 (Max Force)**：權重 **80%** (0.80) — *主要力量強度指標*
* **站立時間 (Stance Time)**：權重 **5%** (0.05) — *肢體承重時間對稱性*
* **步幅時間 (Stride Time)**：權重 **5%** (0.05) — *步態週期對稱性*
* **步幅長度 (Stride Length)**：權重 **5%** (0.05) — *空間位移對稱性*
* **步幅速度 (Stride Velocity)**：權重 **5%** (0.05) — *行進動量對稱性*

### 2. 不對稱分數公式 (Asymmetry Score Formula)
對於每個指標 $m$，取前肢對 (`Left Front / Right Front`) 與後肢對 (`Left Hind / Right Hind`) 之最大不對稱比率：
$$S_m = \max\left( \max\left(\frac{L}{R}, \frac{R}{L}\right)_{\text{Front}}, \max\left(\frac{L}{R}, \frac{R}{L}\right)_{\text{Hind}} \right)$$

最終個體加權不對稱分數為：
$$\text{Weighted Asymmetry Score} = \frac{0.80 \cdot S_{\text{Max Force}} + 0.05 \cdot S_{\text{Stance Time}} + 0.05 \cdot S_{\text{Stride Time}} + 0.05 \cdot S_{\text{Stride Length}} + 0.05 \cdot S_{\text{Stride Velocity}}}{1.00}$$

### 3. 現行劃分門檻 (Active Thresholds)
依據 `videos/classified_video_new/classification_summary.json` 之設定：
* **Level 1 (Sound, 健康)**：$\text{Weighted Score} < 1.30$ （共 **17 筆**，佔 43.6%）
* **Level 2 (Medium, 中度/疑似跛行)**：$1.30 \le \text{Weighted Score} < 1.50$ （共 **14 筆**，佔 35.9%）
* **Level 3 (Lame, 嚴重跛行)**：$\text{Weighted Score} \ge 1.50$ （共 **8 筆**，佔 20.5%）

---

## 二、 全量 Dataset 豬隻分類明細表 (Full Classified Dataset - 39 Pigs)

以下為根據現行標準分類產出之 39 隻豬隻完整數據列表：

| 豬隻編號 (ID) | 分類等級 (Level) | 加權不對稱分數 | 主導指標 (Dominant Metric) | 主導指標單項分數 | 主導肢體對 (Dominant Section) | 兩側偏差解讀 (Interpretation) |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| `1118004` | **Level 1 (Sound)** | `1.0452` | Max Force | `1.0204` | Left Front / Right Front | Right Front higher than Left Front |
| `1118011` | **Level 1 (Sound)** | `1.1914` | Max Force | `1.1905` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1118020` | **Level 1 (Sound)** | `1.2325` | Max Force | `1.2800` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1118023` | **Level 1 (Sound)** | `1.1040` | Max Force | `1.1236` | Left Front / Right Front | Right Front higher than Left Front |
| `1118029` | **Level 1 (Sound)** | `1.1237` | Max Force | `1.1364` | Left Front / Right Front | Right Front higher than Left Front |
| `1118030` | **Level 1 (Sound)** | `1.2504` | Max Force | `1.2987` | Left Front / Right Front | Right Front higher than Left Front |
| `1209043` | **Level 1 (Sound)** | `1.0794` | Max Force | `1.0753` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209045` | **Level 1 (Sound)** | `1.2583` | Max Force | `1.2987` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209047` | **Level 1 (Sound)** | `1.3119` | Max Force | `1.3699` | Left Front / Right Front | Right Front higher than Left Front |
| `1209049` | **Level 1 (Sound)** | `1.2142` | Max Force | `1.2658` | Left Front / Right Front | Right Front higher than Left Front |
| `1209055` | **Level 1 (Sound)** | `1.2892` | Max Force | `1.3000` | Left Front / Right Front | Left Front higher than Right Front |
| `1209056` | **Level 1 (Sound)** | `1.1671` | Max Force | `1.1905` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209065` | **Level 1 (Sound)** | `1.3190` | Max Force | `1.3600` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209068` | **Level 1 (Sound)** | `1.1493` | Max Force | `1.1765` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209076` | **Level 1 (Sound)** | `1.3390` | Max Force | `1.4000` | Left Front / Right Front | Left Front higher than Right Front |
| `1209080` | **Level 1 (Sound)** | `1.1597` | Max Force | `1.1765` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209083` | **Level 1 (Sound)** | `1.3029` | Max Force | `1.3600` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1118008` | **Level 2 (Medium)** | `1.4041` | Max Force | `1.3889` | Left Front / Right Front | Right Front higher than Left Front |
| `1118013` | **Level 2 (Medium)** | `1.4254` | Max Force | `1.5152` | Left Front / Right Front | Right Front higher than Left Front |
| `1118017` | **Level 2 (Medium)** | `1.3660` | Max Force | `1.4400` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209052` | **Level 2 (Medium)** | `1.4867` | Max Force | `1.5873` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209054` | **Level 2 (Medium)** | `1.3626` | Max Force | `1.4493` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209059` | **Level 2 (Medium)** | `1.4150` | Max Force | `1.4900` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209060` | **Level 2 (Medium)** | `1.3968` | Max Force | `1.4800` | Left Front / Right Front | Left Front higher than Right Front |
| `1209062` | **Level 2 (Medium)** | `1.4264` | Max Force | `1.4700` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209063` | **Level 2 (Medium)** | `1.4525` | Max Force | `1.5400` | Left Front / Right Front | Left Front higher than Right Front |
| `1209066` | **Level 2 (Medium)** | `1.3924` | Max Force | `1.4706` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209079` | **Level 2 (Medium)** | `1.4546` | Max Force | `1.5385` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209085` | **Level 2 (Medium)** | `1.4730` | Max Force | `1.5873` | Left Front / Right Front | Right Front higher than Left Front |
| `1209086` | **Level 2 (Medium)** | `1.4550` | Max Force | `1.5625` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209087` | **Level 2 (Medium)** | `1.4815` | Max Force | `1.5873` | Left Front / Right Front | Right Front higher than Left Front |
| `1118006` | **Level 3 (Lame)** | `1.5513` | Max Force | `1.6129` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1118021` | **Level 3 (Lame)** | `2.2444` | Max Force | `2.3810` | Left Front / Right Front | Right Front higher than Left Front |
| `1118034` | **Level 3 (Lame)** | `1.6456` | Max Force | `1.7241` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209038` | **Level 3 (Lame)** | `1.5061` | Max Force | `1.5152` | Left Hind / Right Hind | Right Hind higher than Left Hind |
| `1209039` | **Level 3 (Lame)** | `1.6578` | Max Force | `1.7800` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209061` | **Level 3 (Lame)** | `1.6592` | Max Force | `1.7900` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209073` | **Level 3 (Lame)** | `1.5502` | Max Force | `1.6800` | Left Hind / Right Hind | Left Hind higher than Right Hind |
| `1209075` | **Level 3 (Lame)** | `1.8681` | Max Force | `2.0700` | Left Hind / Right Hind | Left Hind higher than Right Hind |

---

## 三、 SVM 機器學習模型訓練與驗證結果 (SVM Model Validation)

利用 DeepLabCut (DLC) 視覺影像提取之運動學特徵，針對上述分類標籤執行 SVM 模型訓練與 Leave-One-Out (LOO) 交叉驗證：

### 1. 模型混淆矩陣 (Confusion Matrix)


### 2. 模型核心參數與績效
* **最佳超參數配置**：Linear Kernel, $C = 100$, Feature Selection $k = 3$
* **總體 LOO 交叉驗證準率 (Accuracy)**：**53.8% (21/39 正確)**
* **CV F1-Macro**：**0.667**

### 3. 各類別 Precision / Recall / F1-Score
* **Level 1 (Sound, 17 筆)**: Precision = **0.63** | Recall = **0.71** | **F1 = 0.67** (正確預測 12/17)
* **Level 2 (Medium, 14 筆)**: Precision = **0.53** | Recall = **0.57** | **F1 = 0.55** (正確預測 8/14)
* **Level 3 (Lame, 8 筆)**: Precision = 0.20 | Recall = 0.12 | **F1 = 0.15** (正確預測 1/8)
