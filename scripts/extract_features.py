import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import WhisperProcessor

def extract():
    data_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data"
    audio_dir = os.path.join(data_dir, "Audios")
    feat_dir = os.path.join(data_dir, "Whisper_Features_30s") 
    os.makedirs(feat_dir, exist_ok=True)

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.')]

    print(f"開始提取 {len(audio_files)} 筆音訊特徵 (Whisper 原生 30 秒規格)...")

    for filename in tqdm(audio_files):
        save_path = os.path.join(feat_dir, filename.replace('.wav', '.pt'))
        if os.path.exists(save_path): continue 

        try:
            path = os.path.join(audio_dir, filename)
            import librosa
            speech, sr = librosa.load(path, sr=16000)
            
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
            
            torch.save(inputs.half(), save_path)
        except Exception as e:
            print(f"{filename} 失敗: {e}")

if __name__ == "__main__":
    extract()