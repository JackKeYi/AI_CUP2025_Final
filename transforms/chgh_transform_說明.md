# chgh_transform.py 程式碼邏輯說明

## 概述
`chgh_transform.py` 定義了用於處理 CT 影像資料的資料轉換（transform）流程，主要用於醫學影像分割任務。該檔案提供了三種不同的轉換流程：訓練時（train）、驗證時（val）和推理時（inference）。

---

## 一、訓練時轉換流程 (`get_train_transform`)

### 處理流程（按順序執行）

#### 1. **載入影像資料** (`LoadImaged`)
- **功能**：從檔案路徑載入 CT 影像和標籤（label）
- **處理對象**：`["image", "label"]`
- **說明**：讀取原始影像檔案（如 NIfTI 格式）

#### 2. **添加通道維度** (`AddChanneld`)
- **功能**：為影像和標籤添加通道維度
- **處理對象**：`["image", "label"]`
- **說明**：將 3D 影像從 `(H, W, D)` 轉換為 `(1, H, W, D)`

#### 3. **統一影像方向** (`Orientationd`)
- **功能**：將影像方向統一為 RAS（Right-Anterior-Superior）座標系統
- **處理對象**：`["image", "label"]`
- **參數**：`axcodes="RAS"`
- **說明**：確保所有影像使用相同的空間方向，避免因掃描方向不同造成的問題

#### 4. **調整空間解析度** (`Spacingd`)
- **功能**：將影像重新採樣到指定的空間解析度
- **處理對象**：`["image", "label"]`
- **參數**：
  - `pixdim=(args.space_x, args.space_y, args.space_z)`：目標體素間距
  - `mode=("bilinear", "nearest")`：影像使用雙線性插值，標籤使用最近鄰插值
- **說明**：統一不同影像的體素大小，影像使用平滑插值，標籤保持離散值

#### 5. **強度值標準化** (`ScaleIntensityRanged`)
- **功能**：將 CT 影像的強度值（HU 值）縮放到指定範圍
- **處理對象**：`["image"]`（僅影像，不處理標籤）
- **參數**：
  - `a_min`, `a_max`：原始強度值範圍（如 -42 到 423 HU）
  - `b_min`, `b_max`：目標強度值範圍（通常 0 到 1）
  - `clip=True`：將超出範圍的值裁剪
- **說明**：標準化 CT 值，使其適合神經網路訓練

#### 6. **隨機裁剪** (`RandCropByPosNegLabeld`)
- **功能**：根據標籤的正負樣本比例進行隨機裁剪
- **處理對象**：`["image", "label"]`
- **參數**：
  - `spatial_size=(args.roi_x, args.roi_y, args.roi_z)`：裁剪區域大小（如 128×128×128）
  - `pos=4, neg=1`：正樣本與負樣本的比例為 4:1
  - `num_samples=args.num_samples`：每個樣本生成的裁剪數量
- **說明**：確保訓練資料包含足夠的正樣本（有標籤的區域），提高模型學習效果

#### 7. **隨機翻轉** (`RandFlipd`)
- **功能**：隨機沿指定軸翻轉影像
- **處理對象**：`["image", "label"]`
- **參數**：
  - `spatial_axis=[0, 1, 2]`：可在 X、Y、Z 三個軸上翻轉
  - `prob=0.10`：翻轉機率為 10%
- **說明**：資料增強，增加資料多樣性，提高模型泛化能力

#### 8. **隨機旋轉 90 度** (`RandRotate90d`)
- **功能**：隨機將影像旋轉 90 度的倍數
- **處理對象**：`["image", "label"]`
- **參數**：
  - `prob=0.10`：旋轉機率為 10%
  - `max_k=3`：最多旋轉 3 次（270 度）
- **說明**：資料增強，增加空間變化的多樣性

#### 9. **隨機仿射變換** (`RandAffined`)
- **功能**：隨機進行旋轉、縮放等仿射變換
- **處理對象**：`["image", "label"]`
- **參數**：
  - `prob=0.2`：變換機率為 20%
  - `rotate_range=(π/12, π/12, π/12)`：各軸旋轉範圍約 ±15 度
  - `scale_range=(0.1, 0.1, 0.1)`：縮放範圍 ±10%
  - `mode=("bilinear", "nearest")`：影像平滑插值，標籤最近鄰
  - `padding_mode="border"`：邊界填充模式
- **說明**：進階資料增強，模擬真實的影像變形

#### 10. **隨機高斯雜訊** (`RandGaussianNoised`)
- **功能**：向影像添加隨機高斯雜訊
- **處理對象**：`["image"]`（僅影像）
- **參數**：
  - `prob=0.1`：添加雜訊機率為 10%
  - `mean=0.0, std=0.1`：雜訊均值為 0，標準差為 0.1
- **說明**：模擬真實掃描中的雜訊，提高模型對雜訊的魯棒性

#### 11. **隨機高斯平滑** (`RandGaussianSmoothd`)
- **功能**：對影像進行隨機高斯平滑
- **處理對象**：`["image"]`（僅影像）
- **參數**：
  - `sigma_x/y/z=(0.5, 1.0)`：各軸的高斯核標準差範圍
  - `prob=0.1`：平滑機率為 10%
- **說明**：模擬不同解析度的掃描效果

#### 12. **隨機對比度調整** (`RandAdjustContrastd`)
- **功能**：隨機調整影像對比度
- **處理對象**：`["image"]`（僅影像）
- **參數**：
  - `gamma=(0.5, 1.5)`：伽馬值範圍，控制對比度
  - `prob=0.2`：調整機率為 20%
- **說明**：模擬不同掃描參數下的對比度變化

#### 13. **隨機強度偏移** (`RandShiftIntensityd`)
- **功能**：隨機調整影像整體亮度
- **處理對象**：`["image"]`（僅影像）
- **參數**：
  - `offsets=0.1`：偏移量範圍
  - `prob=0.2`：偏移機率為 20%
- **說明**：模擬不同掃描條件下的亮度變化

#### 14. **轉換為張量** (`ToTensord`)
- **功能**：將 NumPy 陣列轉換為 PyTorch 張量
- **處理對象**：`["image", "label"]`
- **說明**：準備資料供 PyTorch 模型使用

---

## 二、驗證時轉換流程 (`get_val_transform`)

驗證流程較簡單，**不包含資料增強**，僅進行必要的預處理：

1. **LoadImaged**：載入影像和標籤
2. **AddChanneld**：添加通道維度
3. **Orientationd**：統一方向為 RAS
4. **Spacingd**：調整空間解析度
5. **ScaleIntensityRanged**：強度值標準化
6. **ToTensord**：轉換為張量

**注意**：驗證時不進行裁剪、翻轉、旋轉等資料增強操作，確保評估結果的一致性。

---

## 三、推理時轉換流程 (`get_inf_transform`)

推理流程與驗證類似，但支援動態的 keys 參數：

1. **LoadImaged**：載入指定 keys 的資料
2. **AddChanneld**：添加通道維度（執行兩次）
3. **Orientationd**：統一方向為 RAS
4. **Spacingd**：調整空間解析度（根據 keys 數量自動設定插值模式）
5. **ScaleIntensityRanged**：僅對 'image' key 進行強度標準化（允許缺少該 key）
6. **ToTensord**：轉換為張量

---

## 四、標籤轉換流程 (`get_label_transform`)

專門用於處理標籤資料：

1. **LoadImaged**：載入標籤檔案
2. **EnsureChannelFirstd**：確保通道維度在前（處理無通道維度的情況）
3. **Orientationd**：統一方向為 RAS
4. **AddChanneld**：添加通道維度
5. **ToTensord**：轉換為張量

---

## 五、在每個 Trial 中的處理流程

根據 `tune.py` 的程式碼，每個 trial 的處理流程如下：

### Trial 執行流程

1. **初始化階段**：
   - 根據 `tune_mode` 設定不同的超參數組合
   - 建立對應的目錄結構（model_dir, log_dir, eval_dir）

2. **訓練階段** (`test_mode=False`)：
   - 使用 `get_train_transform(args)` 處理訓練資料
   - 使用 `get_val_transform(args)` 處理驗證資料
   - 每個訓練樣本會經過完整的訓練轉換流程（包含所有資料增強）

3. **測試階段** (`test_mode=True`)：
   - 使用 `get_val_transform(args)` 或 `get_inf_transform()` 處理測試資料
   - 不進行資料增強，確保結果可重現

### 每個 Trial 中的資料處理特點

- **訓練時**：每個 epoch 中，同一個樣本經過不同的隨機增強（翻轉、旋轉、雜訊等），產生不同的變體
- **驗證/測試時**：每個樣本只處理一次，不進行隨機增強，確保結果一致性
- **空間解析度**：根據 `args.space_x/y/z` 統一所有影像的體素大小
- **強度範圍**：根據 `args.a_min/a_max` 和 `args.b_min/b_max` 標準化 CT 值
- **裁剪大小**：根據 `args.roi_x/y/z` 和 `args.num_samples` 決定訓練時的裁剪策略

### 關鍵參數說明

- `space_x/y/z`：目標體素間距（mm），影響影像解析度
- `a_min/a_max`：原始 CT 值範圍（HU 值）
- `b_min/b_max`：標準化後的目標範圍
- `roi_x/y/z`：訓練時裁剪的區域大小
- `num_samples`：每個樣本生成的裁剪數量

---

## 六、資料增強策略總結

### 空間增強（同時應用於影像和標籤）
- 隨機翻轉（10% 機率）
- 隨機 90 度旋轉（10% 機率）
- 隨機仿射變換（20% 機率）

### 強度增強（僅應用於影像）
- 隨機高斯雜訊（10% 機率）
- 隨機高斯平滑（10% 機率）
- 隨機對比度調整（20% 機率）
- 隨機強度偏移（20% 機率）

### 採樣策略
- 基於標籤的正負樣本平衡裁剪（pos:neg = 4:1）

---

## 七、注意事項

1. **標籤處理**：標籤（label）在空間變換時使用 `nearest` 插值，保持離散值不變
2. **影像處理**：影像在空間變換時使用 `bilinear` 插值，保持平滑
3. **通道維度**：所有轉換都確保資料具有正確的通道維度
4. **方向統一**：所有資料統一為 RAS 方向，確保一致性
5. **強度標準化**：僅對影像進行強度標準化，標籤保持原始值

---

## 八、使用範例

```python
# 在 DataLoader 中使用
from transforms.chgh_transform import get_train_transform, get_val_transform

train_transform = get_train_transform(args)  # 訓練時使用
val_transform = get_val_transform(args)      # 驗證時使用

# 應用轉換
data = train_transform(data_dict)  # data_dict 包含 'image' 和 'label' 路徑
```

---

此文件說明了 `chgh_transform.py` 如何系統性地處理 CT 影像資料，從載入到最終的張量轉換，以及在訓練和驗證階段的不同處理策略。

