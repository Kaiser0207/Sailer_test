# SAILER V2 Architecture & Training Optimization Plan

這份計畫書詳細規劃了將 SAILER 模型從「初階訓練腳本」升級至「資深級機器學習框架」的全部工程改動。依照您的指示，我們**暫時不使用 OmegaConf**，而是採取最經典、穩定的 `configs/default_config.json` 結合內建 `json` 與 `argparse` 模組來達成參數抽離。

## User Review Required
> [!IMPORTANT]
> 請審閱以下計畫，確認架構的改動符合您的預期，尤其是加入了「Sanity Check (乾跑驗證)」和「Dev 廢棄集合併」的大膽策略。若無異議，請給我核准信號，我將立刻為您實作。

## Proposed Changes

---

### Configurations (新建立)

#### [NEW] `configs/default_config.json`
- 建立一個獨立的 JSON 檔案，統一收納所有硬編碼：
  - `data_dir`
  - 訓練超參數 (`epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_ratio`)
  - 模型參數 (`num_classes`, `secondary_class_num`, `dropout_rate`)

---

### Dataset & Data Loading

#### [MODIFY] `src/msp_dataset.py`
- **論文還原 (隱藏招式)**：原本第 132 行只收集了 `Train` 分割裡面的 `no_agree`。我們將改寫為，連同 `Development` 裡面的 `no_agree / other` 一併抓出來，暴力填入 `Train` 樣本池中，徹底壓榨資料集剩餘價值。
- *(註: 經過詳細盤查，論文提到的 Distribution Re-weighting (1/q) 行為，您之前的版本其實已經在 `_get_target_distribution` 及 `w_norm` 實作完成了！這個邏輯非常精美，我們將予保留而不破壞它。)*

---

### Model Architecture (效能與表現)

#### [MODIFY] `src/sailer_model.py`
- **GPU 向量化張量遮罩 (Vectorization)**：
  - 移除 `forward` 中的 `for` 迴圈與 `.item()`。
  - 改用 `mask = torch.arange(max_len).expand(batch, max_len) < lengths.unsqueeze(1)` 搭配 `mask.unsqueeze(-1).float()` 進行矩陣相乘，徹底解除 CPU 堵塞瓶頸。
- **L2 特徵正規化 (L2 Normalization)**：
  - 在執行 `torch.cat([s_emb, t_emb], dim=-1)` 前，強制加入 `F.normalize(s_emb, p=2, dim=-1)` 與 `F.normalize(t_emb, p=2, dim=-1)`，解決 RoBERTa (1024D) 對 Whisper (256D) 的數值輾壓。

---

### Training Pipeline (總司令部)

#### [MODIFY] `train.py`
- **導入 Config 系統**：使用 `argparse` 讀取 `--config configs/default_config.json`。
- **正規亂數套件**：移除手寫的 `seed_everything`，直接使用業界標配 `from transformers import set_seed`。
- **學習率排程器 (LR Scheduler & WD)**：
  - `AdamW` 加入 `weight_decay=1e-4`。
  - 導入 `get_cosine_schedule_with_warmup`，設定總步數前 10% 為暖機期。
- **防呆乾跑 (Sanity Check)**：在 Epoch 0 啟動前，強制抽取一筆 Validation Batch 過一次前向傳播 (Forward Pass)，預先攔截 OOM 或 Shape Mismatch 錯誤。

---

### Documentation

#### [NEW] `CHANGELOG.md`
- 建立標準的版本追蹤日誌，記錄 V1 -> V2 的所有重要架構升級 (Scheduler, Vectorization, JSON Configs)。

#### [NEW] `README.md`
- 產出一份展示於 GitHub 的首頁說明檔，記錄如何執行 `train.py --config configs/default_config.json` 及環境依賴說明。

## 檢查與驗證 (Verification Plan)
1. 執行 `python train.py`，觀察 Sanity Check 是否會噴出提示音並成功秒殺一個 Batch。
2. 觀察前三個 Epochs 的 `val_loss`，確認加入 Warmup 與 L2 之後，開局那種直衝 1.85 的震盪是否已被成功壓制。
