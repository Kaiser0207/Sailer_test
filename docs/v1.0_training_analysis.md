# SAILER 初版模型訓練分析報告 (V1 Training Analysis)

**紀錄日期**：2026-03-29
**實驗目標**：建立基於 Whisper 與 RoBERTa 的雙模態情感辨識基礎模型 (Best Single System Baseline)。
**分析對象**：初版訓練迴圈 (Constant LR) 產生的 Loss 與 Macro-F1 曲線圖。

---

## 🔬 1. 實驗參數設定 (Parameters Used)

本實驗採用以下寫死的初始參數進行 35k Steps 的微調：

- **Epochs**: 15
- **Batch Size**: 64
- **Learning Rate**: `5e-4` (固定學習率，無衰減、無暖機)
- **Optimizer**: AdamW
- **Dropout Rate (SAILER)**: 0.2
- **語音編碼器**: `openai/whisper-large-v3` (Encoder Only, 維度 256)
- **語義編碼器**: `roberta-large` (全 24 層加權平均, 維度 1024)
- **Loss Function**: KLDivLoss (主情緒) + KLDivLoss (次情緒) + MSELoss (AVD連續值) 

---

## 📉 2. 觀測現象 (Observed Phenomena)

在此版實驗中，模型碰到了極為經典的「大模型微調不穩定」現象：

### 現象 A：完美的內部記憶能力
- **`train_loss` 平滑下降**：從 1.83 穩定收斂至 1.45。
- **結論**：神經網路架構編寫正確，前向傳播 (Forward Pass) 與梯度計算正常，模型具備學習資料庫規律的強大能力。

### 現象 B：驚悚的泛化能力崩潰
- **`val_loss` 暴衝與震盪**：在 Step 5k 時不降反升（從 1.73 秒衝 1.85），隨後更是呈現如懸崖般的巨幅震盪。
- **結論**：泛化能力 (Generalization) 在訓練初期就遭到極大破壞，模型在驗證集上陷入無方向的猜測。

### 現象 C：無法收斂的指標
- **`val_macro_f1` 鋸齒狀邊界**：數值在 0.345 到 0.388 之間呈高度不規則反覆彈跳。
- **結論**：優化器無法帶領模型找到 Loss 谷底，在局部最佳解附近反覆橫跳。

---

## 🩺 3. 病灶診斷與根本原因 (Root Causes Diagnostics)

基於上述曲線，診斷出四大核心的架構級病灶：

> [!WARNING]
> **病因一：學習率衝擊 (Learning Rate Shock)**
> - **詳情**：Transformer 類架構 (RoBERTa) 極度脆弱。直接以 `5e-4` 這種對大模型來說「巨大無比」的學習率灌入融合層，會瞬間破壞原本良好的預訓練次空間結構。這正是 `val_loss` 開局暴增的主因。
> - **解方**：必須導入 **Warmup (暖機機制) 與 Cosine Scheduler (餘弦衰減排程器)**。前 10% 步數輕柔拉升，後期穩健煞車。

> [!CAUTION]
> **病因二：未處理的模態特徵輾壓 (Unnormalized Feature Domination)**
> - **詳情**：RoBERTa (1024維) 的向量絕對數值範圍遠大於 Whisper (256維)。直接使用 `torch.cat` 拼接會導致神經網路的梯度過度偏袒文字特徵，演變成「重文輕武」的單模態依賴，失去雙重感官的穩定性。
> - **解方**：在特徵融合前實裝 **特定維度的 L2 Normalization (L2 向量歸一化)**。

> [!IMPORTANT]
> **病因三：正規化強度不足 (Under-Regularization)**
> - **詳情**：`train_loss` 長期下降低於 `val_loss` 是典型的 過擬合 (Overfitting) 起手式。模型過度背誦了訓練集的噪聲。
> - **解方**：增強 Dropout 阻斷能力 (從 0.2 提升至 0.3 甚至 0.4)，並在 AdamW 嚴格實作 `weight_decay=1e-4` (權重衰減) 懲罰過大權重。

> [!NOTE]
> **病因四：多工損失尺度失衡 (Loss Scale Unbalance)**
> - **詳情**：主次情緒的 KL Divergence Loss 與連續特徵的 MSE Loss 量級天生不對等。直接 `loss_a + loss_b + loss_c` 的寫法會讓優化器被數值大的 Loss (如同 AVD 誤差) 單方面綁架。
> - **解方**：導入各任務 Loss 的調控權重係數 (e.g., `0.5 * loss_a + ...`)。

---

## 🚀 4. 行動方案 (Action Items for V2)

為了將模型推進至穩定收斂的 V2 階段，已將上述解方正式排入專案的 `task.md` 待辦清單：
1. **實裝 `get_cosine_schedule_with_warmup`**
2. **對 `w_feat` 與 `t_feat` 實施強制 L2 歸一化**
3. **優化 `train.py` 配置，並啟用乾跑防禦 (Sanity Check)**
