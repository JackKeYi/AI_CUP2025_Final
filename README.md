# AI CUP 2025 Fall: Cardiac Muscle Segmentation (Team_9640)

[![Framework](https://img.shields.io/badge/Framework-MONAI_1.2.0-red)](https://monai.io/)

採用基於 **3D DynUNet** 的深度學習架構，並整合 **Ray Tune** 與 **Optuna** 實現自動化超參數優化 (AutoHPO)。針對醫療影像資料稀缺與類別不平衡（如微小鈣化點）的挑戰，我們設計了兩階段優化策略與權重針對性調整機制。

---

## ✨ 核心特色 (Key Features)

* **AutoHPO 架構**：採用 **TPE (Tree-structured Parzen Estimator)** 演算法取代傳統網格搜索，高效尋找全域最佳解。
* **強健訓練機制 (Robust Training)**：
    * **MD5 斷點續訓**：利用參數雜湊 (Content-based Hashing) 自動生成唯一目錄，解決 Colab 連線不穩問題，確保訓練進度不遺失。
    * **動態學習率**：導入 `LinearWarmupCosineAnnealingLR`，結合 Warmup 與餘弦衰減，提升收斂精度。
* **針對性模型設計**：
    * **3D DynUNet**：動態調整網路拓撲，並採用 **Instance Normalization** 以適應小 Batch Size 訓練。
    * **微小結構優化**：第一層不降採樣 (`Stride=[1,1,1]`)，並將 **Focal Loss** 權重設為 **0.9**，強制模型關注鈣化點與瓣膜。

---

## 🛠️ 環境配置 (Environment Setup)

本專案開發與測試於 **Google Colab (Pro)** 環境。

### 1. 系統需求
* **OS**: Linux Ubuntu 22.04
* **Python**: 3.11.13
* **CUDA**: 12.4

### 2. 依賴套件安裝
請在環境中執行以下指令以安裝必要套件：

```bash
# 基礎數值與影像處理
pip install numpy==1.26.4 opencv-python-headless==4.12.0 gdown==4.6.0 ml_collections

# 深度學習框架 (PyTorch 2.6.0 + CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# MONAI 醫療影像核心框架
pip install monai[all]==1.2.0 monailabel==0.8.5

# 分散式運算與自動化優化 (Ray + Optuna)
pip install ray[default]==2.5.0 optuna tensorboardX

# 模型庫與輔助工具
pip install timm
```
## 📂 專案結構 (Project Structure)

詳細目錄說明：

```text
CardiacSegV2/
├── data_utils/          # 資料載入與處理核心
│   ├── dataset.py       # 定義 DataLoader 與 Dataset 類別
│   └── ...
├── expers/              # 實驗執行腳本 (Entry Points)
│   ├── tune.py          # [核心] 訓練與 AutoHPO 主程式，包含 MD5 續訓邏輯
│   ├── infer.py         # [核心] 推論主程式，包含 TTA 實作
│   └── args.py          # 全域參數定義
├── networks/            # 模型架構定義
│   └── network.py       # DynUNet 模型工廠與建構函式
├── runners/             # 執行邏輯封裝
│   ├── tuner.py         # 訓練迴圈 (Forward/Backward/Validation)
│   └── inferer.py       # 推論迴圈 (Sliding Window Inference)
├── transforms/          # 資料增強與前處理
│   └── chgh_transform.py # 定義幾何變換、強度偏移與正規化流程
└── losses/              # 損失函數
    └── loss.py          # DiceFocalLoss 定義
```

## 📊 最佳超參數配置 (Optimal Hyperparameters)
經過兩階段搜尋（廣域探索 -> 精細搜尋），我們鎖定的最終最佳參數如下：
| 參數 (Parameter) | 設定值 (Value) | 說明 |
| :--- | :--- | :--- |
| **Learning Rate** | `7.14148e-4` | 透過 loguniform 搜尋所得。 |
| **Weight Decay** | `1.24538e-4` | 防止過度擬合。 |
| **Feature Size** | `96` | 確保足夠的模型容量以學習複雜特徵。 |
| **Drop Rate** | `0.272775` | 維持高正則化強度。 |
| **Warmup Epochs** | `40` | 初期穩定梯度。 |
| **Lambda Dice** | `0.319327` | 搜尋範圍限縮於 0.3 至 0.45。。 
| **Lambda Focal** | `0.9` | **關鍵設定**：高權重以強化對鈣化點 (CA) 的學習。 