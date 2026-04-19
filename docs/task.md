# SAILER 專案未來優化待辦清單 (Senior Engineering Checklist)

此清單紀錄了將專案推進至「開源等級 (Open Source Standard)」的最後幾項工程制。

## 📍 1. 訓練管線防護機制 (Sanity Check / Dry Run)
- [x] **讓模型先過一遍資料**：在 `train.py` 進入真正的 Epoch 1 訓練迴圈前，強制抽取一個 Batch 的 Validation Data 進行「模擬推論 (Forward Pass)」。
- **目的**：這絕對是資深工程師的絕招。這能提前揪出維度不符 (Shape Mismatch) 或 OOM (記憶體不足) 的問題。否則如果訓練了 5 個小時才跑到 Validation 階段然後當機，會浪費大量時間。

## 📍 2. 配置分離與套件導入 (Configuration & Utils)
- [x] **導入獨立的 Seed 套件**：使用專業套件 (如 `pytorch_lightning.seed_everything` 或 `accelerate.utils.set_seed`) 取代我們自己手寫的版本，增強跨平台穩定性。
- [x] **架構 Config JSON 解析**：建立 `configs/` 資料夾，並使用原生的 `argparse` 與 `json` 以讀取外部的方式動態注入訓練參數，徹底解除硬編碼。

## 📍 3. 模型效能逼近極限 (Performance Tuning)
- [x] **導入學習率排程器 (LR Scheduler)**：加入 Warmup 與 Cosine Decay。
- [x] **加入特徵 L2 正規化 (L2 Normalization)**：在融合層平衡 256 維語音與 1024 維文字。
- [x] **實作向量化張量遮罩 (Vectorization)**：拔除 Python For 迴圈，徹底解放 GPU 運算瓶頸。

## 📍 4. 專案文檔與版本控制 (Documentation & Versioning)
- [x] **建立 `README.md`**：撰寫專業的專案首頁，包含架構圖、環境安裝 (`requirements.txt`) 與一鍵啟動指令。
- [x] **加入 `CHANGELOG.md`**：建立版本更新紀錄檔，專業地追蹤每一次的架構更動與實驗改進。

## 📍 5. SAILER 論文隱藏細節重現 (Paper-specific Methodologies)
- [x] **分佈反向重賦權 (Distribution Re-weighting)**：實作論文中的類別權重機制 $W_i = 1 / q_i$，對 Loss 進行極端不平衡打擊，讓少數情緒的懲罰係數爆增。 (註：已由使用者在 V1 精彩實作於 msp_dataset 之 w_norm 歸一化邏輯)
- [x] **廢棄資料再利用 (Dev Set Unagreed Data Merging)**：修改 `msp_dataset.py`，將 Validation Set 中被標記為 'Other' 或 'No agreement' 且不作為驗證用途的音檔，偷偷合併混入 Training Set 當中，以最大化訓練集規模。
