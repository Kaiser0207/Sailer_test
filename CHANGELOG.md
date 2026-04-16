# SAILER Project Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v3.5] "OfficialFusion" - 2026-04-12
**核心目標：打破語音單模態天花板，融合官方 SOTA 語音特徵**
- **模型替換**：在訓練管線 (`train.py`) 中動態引入官方競賽釋出的 `tiantiaf/whisper-large-v3-msp-podcast-emotion` (經歷過專精情緒的 LoRA 微調)。
- **架構相容修復**：解決了官方模型將最大長度硬閹割成 15 秒 (1500 frames) 導致的 Shape Mismatch 問題，使 `SAILER_Model` 能夠毫無阻礙地吸收官方增強版特徵。
- **純語音 Benchmark 驗證**：建置生產級評估腳本，確認了官方模型在 Test1 的純語音極限為 Macro F1 ~0.406。將此作為 v3.5 多模態融合的「保底分數」並尋求突破。

## [v3.2] "RetroSOTA" - 2026-04-12
**核心目標：解決資料極度不平衡，並推升泛化能力**
- **特徵級資料增強 (Audio Mixing)**：開發特徵空間的「拼接」技術 (Silence / Overlap)，讓少數類別（例如 Fear, Disgust）在訓練池中的曝光度倍增。
- **標註機率丟棄 (Annotation Dropout)**：在 Dataset 層級引入對抗式標註丟棄機制，防止模型過度擬合佔據多數的「強勢情緒投票」。
- **成效突破**：成功讓我們的從零訓練的多模態系統 (Vanilla Whisper + RoBERTa) 在 MSP-Podcast 上觸及 ~0.38 - 0.41 的水準。

## [v3.1] "Soft Focal Loss & AVD" - 2026-04-10
**核心目標：強化少數類別的存活率，引入輔助任務引導模型**
- **輔助維度拉拔**：在 `SAILER_Model` 中全面啟動 AVD (Arousal / Valence / Dominance) 線性預測頭，並引入比例控制 (`avd_weight: 1.0` ~ `8.0`) 作為優化指標。
- **損失函數升級**：將原先的 KLDivLoss 更換成 `SoftFocalLoss`，讓模型能在後期訓練中無視那些「已經學會的簡單樣本 (Happy/Neutral)」，將算力全數用於攻克困難樣本 (Fear/Disgust/Contempt)。
- **指標升級**：改變驗證迴圈中的關注點，將 **Min. mAP (少數類別平均精度)** 提升到與 F1 Score 同等重要的早停把關地位。

## [v3.0] "Multimodal Foundation" - 2026-04-01
**核心目標：建立最初的多模態基石**
- 發明了 `SAILER_Model` 架構，利用 Cross-Attention 達成 Whisper (語音) 與 RoBERTa (文字) 特徵的跨模態特徵融合。
- 開發完整、支援斷點續傳且具備原子存檔防護的 `train.py` 與 `ExperimentTracker` 訓練監控管線。
- 引入 Weights & Biases 服務 (W&B) 以進行雲端 Loss 曲線與混淆矩陣的即時監控。

## [V2.0.0] - 2026-03-29
### Added
- **Config Separation System**: Implemented `argparse` and JSON configuration (`configs/default_config.json`) to control Hyperparameters externally.
- **Learning Rate Scheduler**: Added `get_cosine_schedule_with_warmup` and `weight_decay=1e-4` to combat early gradient bouncing and late-stage convergence instability.
- **Sanity Check (Dry Run)**: `train.py` now runs a single batch validation forward pass before Epoch 1 to catch `Shape Mismatch` and OOM errors instantly.
- **Vectorized Masking**: Transformed the slow `for` loop in `sailer_model.py`'s temporal mask pooling into a vectorized Broadcasting op, unlocking GPU utilization.
- **L2 Normalization Fusion**: Enforced `F.normalize(p=2)` on both text and speech embeddings before concatenation to prevent 1024D RoBERTa features from dominating 256D Whisper features.
- **Dev-Set Unagreed Data Merging**: Updated `msp_dataset.py` to fetch "No Agreement" and "Other" audio samples from the Development split and merge them into the Training set for maximal data utilization.

### Changed
- Replaced custom `seed_everything` with HuggingFace `transformers.set_seed` for comprehensive determinism across platforms.
- `train.py` correctly imports Whisper Encoder-Only architecture without `ignore_mismatched_sizes`, retaining perfect Positional Embeddings and dropping unnecessary Decoder VRAM overhead.
