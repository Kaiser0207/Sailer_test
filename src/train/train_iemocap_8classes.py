import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import time
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. 定義標籤字典
# ==========================================
EMOTION_TO_IDX = {
    'Neutral': 0, 'Frustrated': 1, 'Angry': 2, 'Sad': 3,
    'Excited': 4, 'Happy': 5, 'Surprised': 6, 'Fearful': 7
}

# ==========================================
# 2. 定義資料集 (支援 SAILER Audio Mixing & Annotation Dropout)
# ==========================================
class IEMOCAPDataset(Dataset):
    def __init__(self, csv_file, whisper_processor, wavlm_processor, is_train=True):
        self.data_frame = pd.read_csv(csv_file)
        self.whisper_processor = whisper_processor
        self.wavlm_processor = wavlm_processor 
        self.is_train = is_train
        
        # --- 數據增強前置作業 ---
        self.class_indices = {i: [] for i in range(8)}
        for idx, row in self.data_frame.iterrows():
            label_idx = EMOTION_TO_IDX[row['primary_emotion']]
            self.class_indices[label_idx].append(idx)
            
        class_counts = {i: len(idxs) for i, idxs in self.class_indices.items()}
        self.majority_classes = [c for c, count in class_counts.items() if count >= 500]
        self.minority_classes = [c for c, count in class_counts.items() if count < 500]
        
        if self.is_train and len(self.minority_classes) > 0:
            min_counts = np.array([class_counts[c] for c in self.minority_classes])
            weights = 1.0 / min_counts
            self.min_sample_probs = weights / weights.sum()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        base_audio_path = row['audio_name']
        base_label_idx = EMOTION_TO_IDX[row['primary_emotion']]
        
        base_speech, sr = librosa.load(base_audio_path, sr=16000)
        soft_label = np.zeros(8, dtype=np.float32)
        soft_label[base_label_idx] = 1.0
        final_speech = base_speech

        if self.is_train:
            # 🚀 終極拼圖 1：SAILER Annotation Dropout (20% 機率丟棄標籤模糊化)
            if random.random() < 0.2:
                # 將標籤變成平均分佈，強迫模型不要過度自信，提升泛化能力
                soft_label = np.ones(8, dtype=np.float32) / 8.0
            
            # 🚀 終極拼圖 2：SAILER Audio Mixing (50% 機率)
            elif random.random() < 0.5 and len(self.minority_classes) > 0:
                mix_idx = None
                if base_label_idx in self.majority_classes:
                    chosen_min_class = np.random.choice(self.minority_classes, p=self.min_sample_probs)
                    mix_idx = random.choice(self.class_indices[chosen_min_class])
                else:
                    chosen_maj_class = random.choice(self.majority_classes)
                    mix_idx = random.choice(self.class_indices[chosen_maj_class])

                if mix_idx is not None:
                    mix_row = self.data_frame.iloc[mix_idx]
                    mix_speech, _ = librosa.load(mix_row['audio_name'], sr=16000)
                    mix_label_idx = EMOTION_TO_IDX[mix_row['primary_emotion']]

                    if random.random() < 0.5: # Mode 1: 靜音拼接
                        silence_duration = random.uniform(0.5, 2.0)
                        silence_samples = np.zeros(int(16000 * silence_duration), dtype=np.float32)
                        final_speech = np.concatenate([base_speech, silence_samples, mix_speech])
                        soft_label = np.zeros(8, dtype=np.float32)
                        soft_label[base_label_idx] = 0.5
                        soft_label[mix_label_idx] = 0.5
                    else: # Mode 2: 重疊混合
                        alpha = random.uniform(0.4, 0.6)
                        delta_t = random.uniform(0.0, 1.0)
                        delta_samples = int(16000 * delta_t)
                        padded_mix = np.concatenate([np.zeros(delta_samples, dtype=np.float32), mix_speech])
                        max_len = max(len(base_speech), len(padded_mix))
                        base_padded = np.pad(base_speech, (0, max_len - len(base_speech)))
                        mix_padded = np.pad(padded_mix, (0, max_len - len(padded_mix)))
                        final_speech = alpha * base_padded + (1 - alpha) * mix_padded
                        soft_label = np.zeros(8, dtype=np.float32)
                        soft_label[base_label_idx] = alpha
                        soft_label[mix_label_idx] = (1 - alpha)

        # 分別轉換為 Whisper 與 WavLM 所需的特徵格式
        whisper_inputs = self.whisper_processor(final_speech, sampling_rate=16000, return_tensors="pt")
        whisper_features = whisper_inputs.input_features.squeeze(0)
        
        wavlm_inputs = self.wavlm_processor(final_speech, sampling_rate=16000, return_tensors="pt")
        wavlm_features = wavlm_inputs.input_values.squeeze(0)

        wavlm_inputs = self.wavlm_processor(
            final_speech, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding='max_length',
            max_length=160000, 
            truncation=True   
        )
        wavlm_features = wavlm_inputs.input_values.squeeze(0)

        return whisper_features, wavlm_features, torch.tensor(soft_label)

# ==========================================
# 3. 定義神經網路大腦 (🔥 SAILER 雙模態融合架構 🔥)
# ==========================================
class SAILER_Model(nn.Module):
    def __init__(self):
        super(SAILER_Model, self).__init__()
        
        # 骨幹一：Whisper (負責聽懂語意)
        self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.float()
        for param in self.whisper.parameters():
            param.requires_grad = False
            
        # 骨幹二：WavLM (負責聽出語氣、音色、情緒)
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").float()
        for param in self.wavlm.parameters():
            param.requires_grad = False

        # 輕量級特徵處理層 (分別處理兩個骨幹的輸出)
        self.whisper_conv = nn.Conv1d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)
        self.wavlm_conv = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1) # 強制時間維度對齊
        
        # 融合層 (Fusion Layer)：把 256 + 256 拼接起來，變成 512
        self.fc_fusion = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 8)

    def forward(self, x_whisper, x_wavlm):
        with torch.no_grad():
            # 抽出深層特徵
            w_states = self.whisper(x_whisper).last_hidden_state # [Batch, T1, 1280]
            v_states = self.wavlm(x_wavlm).last_hidden_state     # [Batch, T2, 1024]

        # Whisper 分支處理
        w_x = w_states.transpose(1, 2)
        w_x = self.whisper_conv(w_x)
        w_x = self.relu(w_x)
        w_x = self.pool(w_x).squeeze(2) # [Batch, 256]

        # WavLM 分支處理
        v_x = v_states.transpose(1, 2)
        v_x = self.wavlm_conv(v_x)
        v_x = self.relu(v_x)
        v_x = self.pool(v_x).squeeze(2) # [Batch, 256]

        # 雙模態特徵融合 (Concatenation)
        fused_features = torch.cat([w_x, v_x], dim=1) # [Batch, 512]
        
        # 最終輸出
        x = self.fc_fusion(fused_features)
        x = self.relu(x)
        output = self.fc_out(x)
        
        return output

# ==========================================
# 4. 主訓練迴圈
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的硬體: {device}")

    batch_size = 64  
    num_epochs = 20
    learning_rate = 4e-4

    print("載入 雙模態 Processors...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")

    print("準備 DataLoader...")
    train_dataset = IEMOCAPDataset("train_iemocap.csv", whisper_processor, wavlm_processor, is_train=True)
    test_dataset = IEMOCAPDataset("test_iemocap.csv", whisper_processor, wavlm_processor, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("初始化...")
    model = SAILER_Model().to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        
        for i, (w_inputs, v_inputs, soft_labels) in enumerate(train_loader):
            w_inputs = w_inputs.to(device)
            v_inputs = v_inputs.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()
            # 餵入雙模態資料
            outputs = model(w_inputs, v_inputs)

            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, soft_labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 60 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()

        # --- 測試階段 ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for w_inputs, v_inputs, soft_labels in test_loader:
                w_inputs = w_inputs.to(device)
                v_inputs = v_inputs.to(device)
                soft_labels = soft_labels.to(device)
                
                outputs = model(w_inputs, v_inputs)

                _, preds = torch.max(outputs, 1)
                _, true_labels = torch.max(soft_labels, 1) 
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())

        target_names = list(EMOTION_TO_IDX.keys())
        report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        
        correct = sum(1 for x, y in zip(all_preds, all_labels) if x == y)
        acc = correct / len(all_labels)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"\n=== Epoch {epoch+1} 驗證結果 (耗時: {epoch_duration:.2f} 秒) ===")
        print(f"Validation Accuracy: {acc:.4f}")
        print(report)
        print("="*30 + "\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_iemocap_model.pth")
            print(f"發現新高準確率！已儲存 (Best Acc: {best_acc:.4f})")

    total_duration = time.time() - total_start_time
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n訓練全數完成，總共耗時: {int(hours)} 小時 {int(minutes)} 分鐘 {seconds:.2f} 秒")

if __name__ == "__main__":
    main()