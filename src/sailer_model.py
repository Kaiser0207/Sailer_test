import torch
import torch.nn as nn

class SAILER_Model(nn.Module):
    def __init__(
        self,
        whisper_dim=1280,
        roberta_dim=1024,
        num_classes=8,
        dropout_rate=0.1,
    ):
        super(SAILER_Model, self).__init__()

        # 3層 Pointwise Conv（filter size = 256）
        self.speech_conv = nn.Sequential(
            nn.Conv1d(whisper_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU()
        )

        # 2層 MLP 分類器
        self.classifier = nn.Sequential(
            nn.Linear(256 + roberta_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, w_seq, t_seq, t_mask):
        
        # 1. 語音特徵：Pointwise Conv + Temporal Average
        s_out = self.speech_conv(w_seq.transpose(1, 2))     
        s_emb = s_out.mean(dim=-1)                          

        # 2. 文字特徵：Masked Temporal Average
        t_emb = (t_seq * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, 1024]

        # 3. 融合與分類
        fused_emb = torch.cat([s_emb, t_emb], dim=1)      
        logits = self.classifier(fused_emb)
        return logits