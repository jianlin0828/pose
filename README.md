# Pose Classification Evaluation Tool based on MediaPipe

這是一個基於 **MediaPipe Pose** 與 **EfficientNet (Head Model)** 的姿態分類評估工具。
專為解決單鏡頭 2D 姿態估計中常見的 **Z 軸深度崩潰 (Z-axis Collapse)** 與 **鏡像誤判 (Mirror Effect)** 問題而設計。本專案引入了「V-Final-Plus-Plus」多層級邏輯修正策略，在不重新訓練姿態模型的前提下，突破物理限制，實現高準確率的姿態分類。

## ✨ 主要功能

- **Z 軸邏輯修正 (Smart Sign Correction)**：利用頭部方向作為錨點，強制修復 MediaPipe 在 90° 側身時的左右誤判。
- **層級防禦機制 (Hierarchical Defense)**：透過「早期傾斜保護」與「背對門檻拉扯」，防止細微動作被大類別吞噬。
- **非對稱主導權 (Asymmetric Dominance)**：針對 MediaPipe 左側數據壓縮問題，動態調整判定門檻。
- **詳細評估報告**：輸出包含 Prompt、Ground Truth (GT)、Prediction 與詳細角度數據 (Yaw/Pitch/Roll) 的 CSV 報表。

## 🛠️ 環境安裝 (Conda)

本專案使用 Conda 進行環境管理，並提供完整的環境設定檔 `environment.yml`。

1. **複製專案**
   ```bash
   git clone [https://github.com/jianlin0828/pose.git](https://github.com/jianlin0828/pose.git)
   cd pose

2.建立虛擬環境使用 environment.yml 還原執行環境：
  ```Bash
  conda env create -f environment.yml

3.啟用環境
  ```Bash
  # 請將 <env_name> 替換為 environment.yml 中第一行定義的名稱 (通常在檔案最上方 name: 欄位)
  conda activate <env_name>

🚀 使用方法
準備資料
圖片資料夾：存放待測圖片。
Prompt CSV (選用)：若需計算準確率 (Accuracy)，需提供包含 filename 與 prompt 欄位的 CSV 檔。
模型權重：請確保您擁有 EfficientNet 頭部模型的 .pth 權重檔。

## 📥 下載模型權重 (Model Weights)

由於 GitHub 檔案大小限制，本專案使用的模型權重檔未包含在儲存庫中。請依照以下步驟下載：

1. **下載權重檔**：
   請至以下連結下載 `DAD-WildHead-EffNetV2-S-best.pth`：
   - [🔗 點擊這裡下載模型權重 (Google Drive/HuggingFace 連結)](https://huggingface.co/HoyerChou/SemiUHPE/tree/main)

2. **放置檔案**：
   請在專案根目錄下建立 `checkpoints` 資料夾，並將檔案放入其中。目錄結構應如下所示：

   ```text
   Pose-Classification-Tool/
   ├── checkpoints/
   │   └── DAD-WildHead-EffNetV2-S-best.pth  <-- 放在這裡
   ├── data/
   ├── src/
   ├── eval_pose_v2.py
   └── ...




執行評估執行以下指令開始評估：
  ```Bash
  python eval_pose_v2.py \
  --img-dir "data/test_images" \
  --out-dir "output/result" \
  --checkpoint "checkpoints//DAD-WildHead-EffNetV2-S-best.pth" \
  --prompts-file "data/prompts.csv"
