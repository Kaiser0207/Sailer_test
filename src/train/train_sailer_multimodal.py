import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import librosa
import time
import sysm 
import datetime
from transformers import WhisperProcessor, WhisperModel, RobertaTokenizer, RobertaModel
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import warnings
warnings.filterwarnings("ignore")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

EMOTION_TO_IDX = {
    'Neutral': 0, 'Frustrated': 1, 'Angry': 2, 'Sad': 3,
    'Excited': 4, 'Happy': 5, 'Surprised': 6, 'Fearful': 7
}

# --- 🚀 Attention Pooling ---
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        x_t = x.transpose(1, 2) 
        attn_weights = self.attention(x_t).squeeze(-1) 
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)
        fused = torch.sum(x_t * attn_weights, dim=1) 
        return fused

# ==========================================
# 2. 終極跨模態資料集
# ==========================================
class SAILER_Ultimate_Dataset(Dataset):
    def __init__(self, csv_file, whisper_processor, roberta_tokenizer, is_train=True):
        self.df = pd.read_csv(csv_file).dropna(subset=['transcription']).reset_index(drop=True)
        self.whisper_processor = whisper_processor
        self.tokenizer = roberta_tokenizer
        self.is_train = is_train
        
        self.class_indices = {i: [] for i in range(8)}
        for idx, row in self.df.iterrows():
            label_idx = EMOTION_TO_IDX[row['primary_emotion']]
            self.class_indices[label_idx].append(idx)
            
        class_counts = {i: len(idxs) for i, idxs in self.class_indices.items()}
        self.majority_classes = [c for c, count in class_counts.items() if count >= 500]
        self.minority_classes = [c for c, count in class_counts.items() if count < 500]
        
        if self.is_train and len(self.minority_classes) > 0:
            min_counts = np.array([class_counts[c] for c in self.minority_classes])
            weights = 1.0 / (min_counts ** 0.5) 
            self.min_sample_probs = weights / weights.sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_speech, _ = librosa.load(row['audio_name'], sr=16000)
        base_text = str(row['transcription'])
        base_label_idx = EMOTION_TO_IDX[row['primary_emotion']]
        
        final_speech, final_text = base_speech, base_text
        soft_label = torch.zeros(8)
        soft_label[base_label_idx] = 1.0

        if self.is_train:
            if random.random() < 0.2:
                soft_label = torch.ones(8) / 8.0
            elif random.random() < 0.3 and len(self.minority_classes) > 0:
                if base_label_idx in self.majority_classes:
                    chosen_min = np.random.choice(self.minority_classes, p=self.min_sample_probs)
                    mix_idx = random.choice(self.class_indices[chosen_min])
                else:
                    chosen_maj = random.choice(self.majority_classes)
                    mix_idx = random.choice(self.class_indices[chosen_maj])

                mix_row = self.df.iloc[mix_idx]
                mix_speech, _ = librosa.load(mix_row['audio_name'], sr=16000)
                mix_text = str(mix_row['transcription'])
                mix_label_idx = EMOTION_TO_IDX[mix_row['primary_emotion']]

                if random.random() < 0.5: 
                    silence = np.zeros(int(16000 * random.uniform(0.5, 1.2)), dtype=np.float32)
                    final_speech = np.concatenate([base_speech, silence, mix_speech])
                    final_text = f"{base_text} [SEP] {mix_text}"
                    soft_label = torch.zeros(8)
                    soft_label[base_label_idx] = 0.5; soft_label[mix_label_idx] = 0.5
                else: 
                    alpha = random.uniform(0.4, 0.6)
                    max_len = max(len(base_speech), len(mix_speech))
                    final_speech = np.pad(base_speech, (0, max_len-len(base_speech))) * alpha + \
                                   np.pad(mix_speech, (0, max_len-len(mix_speech))) * (1-alpha)
                    final_text = f"{base_text} [SEP] {mix_text}"
                    soft_label = torch.zeros(8)
                    soft_label[base_label_idx] = alpha; soft_label[mix_label_idx] = (1-alpha)

        # 🚨 使用 V3 的 128 維度特徵提取
        w_feat = self.whisper_processor(final_speech, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        t_in = self.tokenizer(final_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        
        return w_feat, t_in.input_ids.squeeze(0), t_in.attention_mask.squeeze(0), soft_label

# ==========================================
# 3. SAILER 論文架構模型 (駭客植入版)
# ==========================================
class SAILER_Ultimate_Model(nn.Module):
    def __init__(self):
        super(SAILER_Ultimate_Model, self).__init__()
        
        self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.float()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        
        try:
            model_path = hf_hub_download(repo_id="tiantiaf/whisper-large-v3-msp-podcast-emotion", filename="model.safetensors")
            state_dict = load_file(model_path)
            
            encoder_weights = {}
            for k, v in state_dict.items():
                if k.startswith("backbone_model.encoder."):
                    new_key = k.replace("backbone_model.encoder.", "")
                    # 略過位置編碼形狀不合的問題，保留原廠位置編碼
                    if new_key == "embed_positions.weight" and v.shape != self.whisper.embed_positions.weight.shape:
                        continue
                    encoder_weights[new_key] = v
                    
            missing, unexpected = self.whisper.load_state_dict(encoder_weights, strict=False)
            print(f"權重植入成功！成功繞過作者的 Config Bug！(載入了 {len(encoder_weights)} 層參數)")
        except Exception as e:
            print(f"權重植入失敗，使用原廠 Whisper: {e}")
        # -----------------------------------------------------

        for p in list(self.whisper.parameters()) + list(self.roberta.parameters()): 
            p.requires_grad = False

        # 🚨 配合 Large 大腦，接管改回 1280
        self.speech_conv = nn.Conv1d(1280, 256, kernel_size=1)
        self.text_conv = nn.Conv1d(1024, 256, kernel_size=1)
        
        self.speech_pool = AttentionPooling(256)
        self.text_pool = AttentionPooling(256)
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 8)

    def forward(self, w_feat, t_ids, t_mask):
        with torch.no_grad():
            s_out = self.whisper(w_feat).last_hidden_state
            t_out = self.roberta(input_ids=t_ids, attention_mask=t_mask).last_hidden_state

        s_x = self.speech_pool(self.relu(self.speech_conv(s_out.transpose(1, 2))))
        t_x = self.text_pool(self.relu(self.text_conv(t_out.transpose(1, 2))), mask=t_mask)
        return self.fc(torch.cat([s_x, t_x], dim=1))

# ==========================================
# 4. 主程式
# ==========================================
# ==========================================
# 4. 主程式
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"啟動 | 硬體: {device}")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"epoch_log_{timestamp}.txt"
    print(f"Epoch 儲存於: {log_filename}")

    batch_size = 64
    lr = 3e-4 
    num_epochs = 20

    csv_path = "/home/brant/Project/SAILER_test/csv_manifests/iemocap_with_text.csv"

    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    full_dataset_len = len(pd.read_csv(csv_path).dropna(subset=['transcription']))
    train_idx, val_idx = train_test_split(list(range(full_dataset_len)), test_size=0.2, random_state=42)
    
    train_dataset = Subset(SAILER_Ultimate_Dataset(csv_path, whisper_processor, roberta_tokenizer, is_train=True), train_idx)
    val_dataset = Subset(SAILER_Ultimate_Dataset(csv_path, whisper_processor, roberta_tokenizer, is_train=False), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SAILER_Ultimate_Model().to(device)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    total_start = time.time()
    best_acc = 0.0

    print(f"\n訓練開始... (預計 Epoch: {num_epochs}, BS: {batch_size}, LR: {lr})")

    # 🚨 開啟檔案寫入模式，準備記錄 Epoch 精華
    with open(log_filename, "a", encoding="utf-8") as f_log:
        f_log.write(f"訓練開始 (Epochs: {num_epochs}, BS: {batch_size}, LR: {lr})\n")
        f_log.write("=" * 60 + "\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # --- 訓練 ---
            model.train()
            train_loss = 0
            for i, (w_f, t_id, t_m, lbl) in enumerate(train_loader):
                w_f, t_id, t_m, lbl = w_f.to(device), t_id.to(device), t_m.to(device), lbl.to(device)
                optimizer.zero_grad()
                
                out = model(w_f, t_id, t_m)
                loss = criterion(F.log_softmax(out, dim=1), lbl)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                if (i+1) % 20 == 0:
                    print(f"E[{epoch+1}] Step[{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            # --- 驗證 ---
            model.eval()
            v_preds, v_labels = [], []
            with torch.no_grad():
                for w_f, t_id, t_m, lbl in val_loader:
                    out = model(w_f.to(device), t_id.to(device), t_m.to(device))
                    v_preds.extend(torch.max(out, 1)[1].cpu().numpy())
                    v_labels.extend(torch.max(lbl, 1)[1].cpu().numpy())

            acc = accuracy_score(v_labels, v_preds)
            report = classification_report(
                v_labels, 
                v_preds, 
                target_names=list(EMOTION_TO_IDX.keys()),
                digits=4,
                zero_division=0
            )

            duration = time.time() - epoch_start
            
            # 1. 印在終端機
            print(f"\nEpoch {epoch+1} 驗證結果 (耗時: {duration:.2f}s)")
            print(f"Accuracy: {acc:.4f}")
            print(report) 
            print("-" * 60)

            # 2. 🚨 寫進 TXT 檔案
            f_log.write(f"\nEpoch {epoch+1} 驗證結果 (耗時: {duration:.2f}s)\n")
            f_log.write(f"Accuracy: {acc:.4f}\n")
            f_log.write(report + "\n")
            f_log.write("-" * 60 + "\n")
            f_log.flush() # 強制立刻寫入硬碟

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_sailer_multimodal_ultimate.pth")
                msg = f"刷新紀錄！已儲存 ({best_acc:.4f})"
                print(msg)
                f_log.write(msg + "\n") # 將破紀錄的訊息也寫入 TXT
                f_log.flush()
            
            scheduler.step()
            print("-" * 60)

        total_time = time.time() - total_start
        final_msg = f"\n總耗時: {total_time/3600:.2f} 小時 | 最高準確率: {best_acc:.4f}"
        print(final_msg)
        f_log.write(final_msg + "\n")

if __name__ == "__main__":
    main()