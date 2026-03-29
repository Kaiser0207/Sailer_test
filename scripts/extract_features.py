import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import WhisperProcessor

def extract():
    data_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data"
    audio_dir = os.path.join(data_dir, "Audios")
    feat_dir = os.path.join(data_dir, "Whisper_Features_15s") 
    os.makedirs(feat_dir, exist_ok=True)

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    consensus_path = os.path.join(data_dir, "Labels", "labels_consensus.csv")
    df = pd.read_csv(consensus_path)
    valid_files = set(df['FileName'].tolist())
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.') and f in valid_files]

    print(f"只提取有標籤的 {len(audio_files)} 筆音訊特徵 (裁剪至 15 秒長度)...")

    for filename in tqdm(audio_files):
        save_path = os.path.join(feat_dir, filename.replace('.wav', '.pt'))
        if os.path.exists(save_path): continue 

        try:
            path = os.path.join(audio_dir, filename)
            import librosa
            speech, sr = librosa.load(path, sr=16000)
            
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
            inputs = inputs[:, :1500]
            
            torch.save(inputs.half(), save_path)
        except Exception as e:
            print(f"{filename} 失敗: {e}")

if __name__ == "__main__":
    extract()