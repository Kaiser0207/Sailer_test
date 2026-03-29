# V2 Architecture Refactoring Walkthrough

本次的架構大升級涵蓋了我們稍早建立的整份 `task.md`，主要目的是將 SAILER 系統自「研究初期」升級為「開源競賽等級」。

## 📌 完成的重大變革 (Major Changes Deployed)

### 1. 外部配置驅動 (Configuration Driven)
- **變動**：移除了 `train.py` 中寫死的矩陣與路徑，全面改用 Python 原生套件 `argparse` 與 `json` 做讀取。
- **優勢**：未來如果要調 LR 或是 Batch Size，只需修改 `configs/default_config.json` 即可。這大幅度降低了改錯程式碼的風險。

### 2. 資料集再榨取 (Dev Set Data Merging)
- **變動**：`msp_dataset.py` 中引進了作者論文的隱藏招式。原本被歸類在 `Validation` 且沒有共識標準答案的 "Other" 樣本，現在會被我們的資料管線自動抓出來、丟進 `Train` 樣本池中擴增。
- **優勢**：在「軟標籤 (Soft Label)」的學習目標下，任何具有人類意見分佈的語音都能幫助模型描繪 Emotion Space。

### 3. 硬體解放與防呆 (Vectorization & Sanity Check)
- **向量化 (Vectorization)**：在 `sailer_model.py` 實作了 **Vectorized Tensor Masking**，利用 `torch.arange` 取代效率低落的 `for` 迴圈來計算音段的均值池化 (Average Pooling)。這完全解放了 CPU，不再卡住 GPU 計算流。
- **乾跑保護 (Sanity Check)**：在 `train.py` 正式進入 `for epoch` 漫長的訓練前，現在會**強迫抽出第一筆 Validation Batch 通過網路**，確認沒問題才放行，攔截各種因為改動導致維度錯亂的意外！

### 4. 收斂穩定性 (Convergence Stability)
- **排程器與正規化**：`AdamW` 優化器全面升格，同時掛載了 `get_cosine_schedule_with_warmup` (前 10% 暖機，後段餘弦滑行) 以及 `weight_decay=1e-4`。
- **特徵單位化 (L2 Normalization)**：利用 `F.normalize(p=2)` 將 1024D 的 RoBERTa 特徵與 256D 的 Whisper 特徵強制壓縮在同一個單位長度內再進行拼接。這將能根除我們先前從曲線上觀察到的「特徵輾壓」與「嚴重震盪」。

## 🔍 如何啟動下一次實驗

架構均已準備妥當！您現在隨時可以開啟終端機，執行這行最專業的起手指令：
```bash
python train.py --config configs/default_config.json
```
