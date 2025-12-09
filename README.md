# AI Cup Cardiac Muscle Segmentation (3D U-Net with AutoHPO)

主要特色：
* **AutoHPO**：基於 TPE 演算法的自動化超參數搜尋。
* **Robust Training**：具備 MD5 雜湊斷點續訓機制，適應 Colab 不穩定環境。
* **Advanced Inference**：實作 Test-Time Augmentation (TTA) 以優化微小結構 (如冠狀動脈) 的分割精度。

---

## 🛠️ 環境配置 (Environment Setup)

本專案設計於 **Google Colab** 環境下執行。

### 1. 軟體與硬體需求
* **Runtime**: Google Colab 
* **Python**: 3.11.13
* **CUDA**: 12.4

### 2. 依賴套件安裝
請在 Colab 的第一個 Cell 執行以下安裝指令，以確保環境版本與我們訓練時一致：

```bash
# 基礎依賴與影像處理
pip install numpy==1.26.4 opencv-python-headless==4.12.0 gdown==4.6.0 ml_collections

# 深度學習框架 (PyTorch 2.6.0 + CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# MONAI 生態系
pip install monai[all]==1.2.0 monailabel==0.8.5

# 分散式運算與優化 (Ray + Optuna)
pip install ray[default]==2.5.0 optuna tensorboardX

# 輔助工具
pip install timm
