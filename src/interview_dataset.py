import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm


class InterviewEmotionDataset(Dataset):
    """
    面試場景 4 類情緒資料集 (Neutral, Happy, Fear, Surprise)。
    直接載入 raw audio wav 檔案，交由官方 WhisperWrapper 內建的 feature_extractor 處理。
    使用 Hard Label + CrossEntropyLoss，不需要 Soft Label / Secondary / AVD。
    """

    # MSP-Podcast 原始標籤 → 4 類新 Index
    EMOTION_MAP_SHORT = {'N': 0, 'H': 1, 'F': 2, 'U': 3}
    EMOTION_MAP_LONG = {'Neutral': 0, 'Happy': 1, 'Fear': 2, 'Surprise': 3}
    EMOTION_NAMES = ['Neutral', 'Happy', 'Fear', 'Surprise']

    def __init__(self, data_dir, split="Train", max_audio_sec=15, sample_rate=16000):
        self.data_dir = data_dir
        self.split = split
        self.max_audio_sec = max_audio_sec
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_sec * sample_rate  # 15s * 16000 = 240000 samples

        self.audio_dir = os.path.join(data_dir, "Audios")
        self.consensus_path = os.path.join(data_dir, "Labels", "labels_consensus.csv")

        self.data_records = self._load_data()

        # 計算各類樣本數（用於 class weights 與 oversampling）
        self.class_counts = np.zeros(4, dtype=np.int64)
        for r in self.data_records:
            self.class_counts[r['label']] += 1

        # 開根號逆頻率 class weights（交給 CrossEntropyLoss 使用）
        # 使用 sqrt(1/freq) 而非 1/freq，避免少數類別的懲罰倍率過度極端
        inv_freq = 1.0 / (self.class_counts + 1e-8)
        sqrt_inv_freq = np.sqrt(inv_freq)
        self.class_weights = torch.tensor(sqrt_inv_freq / sqrt_inv_freq.sum() * 4, dtype=torch.float32)

        # 每筆樣本的採樣權重（交給 WeightedRandomSampler 使用）
        sample_weights = inv_freq[np.array([r['label'] for r in self.data_records])]
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

        print(f"[{split}] 4-class 面試情緒資料集載入完成！")
        for i, name in enumerate(self.EMOTION_NAMES):
            print(f"  {name}: {self.class_counts[i]} 筆")
        print(f"  總計: {len(self.data_records)} 筆")

    def _load_data(self):
        """過濾並載入 4 類面試情緒資料。"""
        records = []
        df = pd.read_csv(self.consensus_path)

        # 根據 Split 過濾
        if self.split == "Train":
            df_use = df[df['Split_Set'] == "Train"]
        elif self.split == "Development":
            df_use = df[df['Split_Set'] == "Development"]
        elif self.split == "Test1":
            df_use = df[df['Split_Set'] == "Test1"]
        elif self.split == "Test2":
            df_use = df[df['Split_Set'] == "Test2"]
        else:
            raise ValueError(f"未知的 split: {self.split}")

        # 只保留 4 個目標情緒
        df_use = df_use[df_use['EmoClass'].isin(self.EMOTION_MAP_SHORT.keys())]

        for _, row in tqdm(df_use.iterrows(), total=len(df_use), desc=f"Scanning {self.split}"):
            filename = row['FileName']
            audio_path = os.path.join(self.audio_dir, filename)

            if not os.path.exists(audio_path):
                continue

            label = self.EMOTION_MAP_SHORT[row['EmoClass']]

            records.append({
                "audio_path": audio_path,
                "label": label,
                "filename": filename,
            })

        return records

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]

        # 載入音訊（使用 soundfile，自動轉換為 16kHz mono）
        waveform, sr = sf.read(record["audio_path"], dtype='float32')

        # 轉 mono (soundfile 回傳的是 [num_samples] 或 [num_samples, channels])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        waveform = torch.tensor(waveform, dtype=torch.float32)  # [num_samples]

        # 截斷至 max_audio_sec
        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]

        label = torch.tensor(record["label"], dtype=torch.long)
        length = torch.tensor(waveform.shape[0], dtype=torch.long)

        return waveform, label, length
