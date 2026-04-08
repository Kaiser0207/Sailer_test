import torch
import torch.nn as nn
import torch.nn.functional as F

class SAILER_Model(nn.Module):
    """
    SAILER 雙模態情感辨識模型 (Speech-Audio Integrated Learning for Emotion Recognition).
    結合了 Whisper (語音) 與 RoBERTa (文字) 萃取的高維度特徵，進行多任務 (Multi-task) 預測。
    
    核心架構：
    1. Speech Encoder: 使用 1D-Conv 將 Whisper 提取的 1280 維聲紋特徵降維至 256 維，過濾冗餘雜訊。
    2. Text Encoder: 對 RoBERTa 全部的 25 層隱藏狀態 (Hidden States) 套用可學習的加權平均 (Learnable Weighted Average)。
    3. Multimodal Fusion: 將 256 維語音特徵與 1024 維文字特徵拼接 (Concatenation) 形成 1280 維的綜合特徵。
    4. Multi-task Heads: 輸出 Primary Emotion (8分類)、Secondary Emotion (17分類) 以及 AVD 三維回歸 (Arousal/Valence/Dominance)。
    """
    def __init__(
        self,
        whisper_dim=1280,
        roberta_dim=1024,
        num_roberta_layers=25,  
        hidden_dim=256,
        num_classes=8,
        secondary_class_num=17,
        dropout_rate=0.1,
    ):
        super(SAILER_Model, self).__init__()

        # =========================================
        # 1. 特徵處理層 (Feature Processing Layers)
        # =========================================

        # RoBERTa 各層融合權重 (Learnable Layer Weights)
        # 初始化為均勻分佈 (1/25)，模型在訓練時會由反向傳播自動學會哪幾層對文字情感辨識最有幫助
        self.text_layer_weights = nn.Parameter(
            torch.ones(num_roberta_layers) / num_roberta_layers
        )

        # 語音特徵降維與過濾網路 (Speech Downstream Network)
        # 利用 3 層 Pointwise Conv (Kernel=1) 扮演 Feature Bottleneck，將肥大的 1280 維輕量化至 256 維
        self.speech_conv = nn.Sequential(
            nn.Conv1d(whisper_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        # ==========================================
        # 2. 多任務輸出頭 (Multi-task Prediction Heads)
        # 標準的 2-layer MLP 架構，負責不同的情感維度預測
        # 融合層維度 (fused_emb) = 語音 hidden_dim(256) + 文字 roberta_dim(1024) = 1280
        # ==========================================

        # 主情緒分類頭 (Primary Emotion: 8 classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        # 輔助次要情緒分類頭 (Secondary Emotion: 17 classes)
        # 包含 8 種主情緒與 9 種 Other 的細分變體，有助於模型學習更極端、細微的情感特徵 (Fine-grained features)
        self.secondary_emotion_layer = nn.Sequential(
            nn.Linear(hidden_dim + roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, secondary_class_num)
        )

        # 喚醒度回歸頭 (Arousal Regression: 激動程度或平緩) -> 輸出範圍 [0, 1]
        self.arousal_layer = nn.Sequential(
            nn.Linear(hidden_dim + roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 正負向回歸頭 (Valence Regression: 情緒正面或負面) -> 輸出範圍 [0, 1]
        self.valence_layer = nn.Sequential(
            nn.Linear(hidden_dim + roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 支配度回歸頭 (Dominance Regression: 氣場強烈或弱勢) -> 輸出範圍 [0, 1]
        self.dominance_layer = nn.Sequential(
            nn.Linear(hidden_dim + roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, w_seq, t_hidden_states, t_mask, lengths=None):
        """
        前向傳播邏輯 (Forward Pass)
        Args:
            w_seq (Tensor):           [B, T_speech, 1280] Whisper 編碼器的輸出特徵矩陣
            t_hidden_states (Tuple):  長度 25 的 Tuple，每個元素為 [B, T_text, 1024] (RoBERTa 每一層的輸出)
            t_mask (Tensor):          [B, T_text] 文字 Attention Mask (用來標記 padding)
            lengths (Tensor):         [B] 每一筆音訊的實際有效長度 (Effective frames)，用於有效時間池化 (Temporal Pooling)
        Returns:
            primary_logits:    [B, 8]  主情緒的分類 logits
            secondary_logits:  [B, 17] 次要情緒的分類 logits
            arousal:           [B, 1]  喚醒度預測值
            valence:           [B, 1]  正負向預測值
            dominance:         [B, 1]  支配度預測值
        """

        # -------------------------------------------------------------------
        # 1. 語音特徵處理 (Speech Feature Processing)
        # 流程: Pointwise Conv 降維 -> Masked Temporal Average Pooling
        # -------------------------------------------------------------------
        s_out = self.speech_conv(w_seq.transpose(1, 2))  

        if lengths is not None:
            B, D, T = s_out.shape
            # 動態有效長度保護 (確保不超過特徵矩陣的最大時間軸)
            clipped_lengths = torch.clamp(lengths, max=T)
            valid_lengths = clipped_lengths.clamp(min=1).unsqueeze(1).float()
            
            # 使用 vectorized masking 消除原有的慢速 for 迴圈
            mask = torch.arange(T, device=s_out.device).unsqueeze(0) < clipped_lengths.unsqueeze(1)
            mask_float = mask.unsqueeze(1).float()  # [B, 1, T]
            
            s_emb = (s_out * mask_float).sum(dim=-1) / valid_lengths  # [B, 256]
        else:
            s_emb = s_out.mean(dim=-1)  # [B, 256]

        # -------------------------------------------------------------------
        # 2. 文字特徵處理 (Text Feature Processing)
        # 流程: Learnable Weighted Average (跨層融合) -> Temporal Average Pooling
        # -------------------------------------------------------------------
        stacked = torch.stack(t_hidden_states, dim=0)  # 將 25 層堆疊組成四維張量: [num_layers, B, T, 1024]
        
        # 對權重參數進行 Softmax，確保加總為 1 且呈現機率分佈
        norm_weights = torch.softmax(self.text_layer_weights, dim=0)  
        
        # 利用廣播機制 (Broadcasting) 執行矩陣加權總和，將 25 層平滑地融合壓縮為單層結果
        t_seq = (norm_weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)  # [B, T, 1024]
        
        # 針對有效文字 Token 進行全局平均 (利用 t_mask 遮罩過濾掉無意義的 [PAD] token，防止雜訊干擾)
        t_emb = (t_seq * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-8)  # 產出終極文字特徵: [B, 1024]

        # -------------------------------------------------------------------
        # 3. 多模態融合與多任務預測 (Multimodal Fusion & Multi-task Prediction)
        # -------------------------------------------------------------------
        # 將語音 (256維) 與文字 (1024維) 直接拼接 (Concatenation)，建立 1280 維的聯合表徵空間
        # 加上 L2 Normalization (迫使向長度單位化)，避免 RoBERTa 的數值規模過大輾壓 Whisper
        s_emb = F.normalize(s_emb, p=2, dim=-1)
        t_emb = F.normalize(t_emb, p=2, dim=-1)
        fused_emb = torch.cat([s_emb, t_emb], dim=1)  # 融合特徵: [B, 1280]

        # 將融合後的特徵分五路，平行運算分發給各自的任務網路
        primary_logits = self.classifier(fused_emb)                  # [B, 8]
        secondary_logits = self.secondary_emotion_layer(fused_emb)   # [B, 17]
        arousal = self.arousal_layer(fused_emb)                      # [B, 1]
        valence = self.valence_layer(fused_emb)                      # [B, 1]
        dominance = self.dominance_layer(fused_emb)                  # [B, 1]

        return primary_logits, secondary_logits, arousal, valence, dominance

class SAILER_Modern_Model(nn.Module):
    """
    SAILER 現代化升級版 (WavLM + ModernBERT)
    1. Speech Encoder: WavLM-Large (輸出 1024 維)。
    2. Text Encoder: ModernBERT-Large (輸出 1024 維，全 28 層加權平均)。
    3. Multimodal Fusion: `fused_emb` 為 1024(Speech) + 1024(Text) = 2048 維。
    """
    def __init__(
        self,
        wavlm_dim=1024,
        modernbert_dim=1024,
        num_text_layers=28,
        hidden_dim=256,
        num_classes=8,
        secondary_class_num=17,
        dropout_rate=0.1,
    ):
        super(SAILER_Modern_Model, self).__init__()

        self.text_layer_weights = nn.Parameter(
            torch.ones(num_text_layers) / num_text_layers
        )

        self.speech_conv = nn.Sequential(
            nn.Conv1d(wavlm_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        fused_dim = hidden_dim + modernbert_dim # 256 + 1024 = 1280

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self.secondary_emotion_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, secondary_class_num)
        )

        self.arousal_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.valence_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dominance_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, w_seq, t_hidden_states, t_mask, lengths=None):
        s_out = self.speech_conv(w_seq.transpose(1, 2))  

        if lengths is not None:
            B, D, T = s_out.shape
            clipped_lengths = torch.clamp(lengths, max=T)
            valid_lengths = clipped_lengths.clamp(min=1).unsqueeze(1).float()
            mask = torch.arange(T, device=s_out.device).unsqueeze(0) < clipped_lengths.unsqueeze(1)
            mask_float = mask.unsqueeze(1).float()  
            s_emb = (s_out * mask_float).sum(dim=-1) / valid_lengths
        else:
            s_emb = s_out.mean(dim=-1)

        stacked = torch.stack(t_hidden_states, dim=0)  
        norm_weights = torch.softmax(self.text_layer_weights, dim=0)  
        t_seq = (norm_weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)  
        t_emb = (t_seq * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-8)  

        s_emb = F.normalize(s_emb, p=2, dim=-1)
        t_emb = F.normalize(t_emb, p=2, dim=-1)
        fused_emb = torch.cat([s_emb, t_emb], dim=1)  

        primary_logits = self.classifier(fused_emb)                  
        secondary_logits = self.secondary_emotion_layer(fused_emb)   
        arousal = self.arousal_layer(fused_emb)                      
        valence = self.valence_layer(fused_emb)                      
        dominance = self.dominance_layer(fused_emb)                  

        return primary_logits, secondary_logits, arousal, valence, dominance