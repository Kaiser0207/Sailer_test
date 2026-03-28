import os
import re
import sys
import torch
import torch.nn as nn
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import WhisperProcessor, RobertaTokenizer, WhisperModel, RobertaModel, pipeline

# ==========================================
# 1. Configuration & Label Mapping
# ==========================================
EMOTION_TO_IDX = {
    'Neutral': 0, 'Frustrated': 1, 'Angry': 2, 'Sad': 3,
    'Excited': 4, 'Happy': 5, 'Surprised': 6, 'Fearful': 7
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

# Standard IEMOCAP labels (lowercase for easy matching)
VALID_EMOTIONS_LOWER = {k.lower(): k for k in EMOTION_TO_IDX.keys()}

# Custom Folder Mapping: 將朋友自訂的資料夾名稱，強制對應到 IEMOCAP 的標準標籤
# 若朋友生成了 disgust (無法對應)，系統會自動標記為 Out-of-Distribution (OOD) 不計入準確率
FOLDER_NAME_MAPPING = {
    "fear": "Fearful",
    "peaceful": "Neutral",
    "depressed": "Sad",
    "surprise": "Surprised"
}

# ==========================================
# 2. Model Architecture
# ==========================================
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
        return torch.sum(x_t * attn_weights, dim=1) 

class SAILER_Ultimate_Model(nn.Module):
    def __init__(self):
        super(SAILER_Ultimate_Model, self).__init__()
        self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.float()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
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
# 3. Main Inference & Evaluation Routine
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Initializing inference engine on {device}...")

    # Parameters
    test_folder = "/home/brant/Project/SAILER_test/output_multi_text20" 
    weight_path = "best_sailer_multimodal_ultimate.pth" 

    print("[System] Loading feature extractors and model weights...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    model = SAILER_Ultimate_Model().to(device)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"[Error] Weight file '{weight_path}' not found.")
        sys.exit(1)

    print("[System] Initializing Automatic Speech Recognition (ASR) pipeline...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=0 if torch.cuda.is_available() else -1
    )
    print("[System] Initialization complete.")

    # 掃描音檔
    valid_extensions = ('.wav', '.mp3', '.flac', '.m4a')
    audio_files = []
    for root, dirs, files in os.walk(test_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                audio_files.append(os.path.join(root, filename))

    if not audio_files:
        print("[Warning] No valid audio files found in the specified directory.")
        sys.exit(0)

    print(f"\n[Info] Processing {len(audio_files)} audio files...")

    results = []
    y_true = []
    y_pred = []

    # Inference Loop
    for audio_path in tqdm(audio_files, desc="Inference Progress", unit="file"):
        path_parts = audio_path.split(os.sep)
        filename = path_parts[-1]       
        target_emotion_raw = path_parts[-2] 
        text_group = path_parts[-3]     

        # Alpha Extraction
        alpha_match = re.search(r'alpha([0-9.]+)', filename)
        alpha_value = alpha_match.group(1) if alpha_match else "unknown"

        # Step 1: ASR Transcription
        try:
            transcription_result = asr_pipeline(audio_path)
            auto_text = transcription_result["text"].strip()
            if not auto_text: 
                auto_text = "I don't know what to say."
        except Exception:
            auto_text = "Transcription failed."

        # Step 2: Audio Loading
        try:
            speech, _ = librosa.load(audio_path, sr=16000)
        except Exception:
            continue
        
        # Step 3: Feature Preparation
        w_feat = whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features
        t_in = roberta_tokenizer(auto_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        
        w_feat = w_feat.to(device)
        t_ids = t_in.input_ids.to(device)
        t_mask = t_in.attention_mask.to(device)
        
        # Step 4: Emotion Prediction
        with torch.no_grad():
            out = model(w_feat, t_ids, t_mask)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            
        pred_idx = probs.argmax()
        pred_emotion = IDX_TO_EMOTION[pred_idx]
        
        # Step 5: Ground Truth Mapping (for Accuracy calculation)
        target_emotion_lower = target_emotion_raw.lower()
        true_label = None
        
        if target_emotion_lower in FOLDER_NAME_MAPPING:
            true_label = FOLDER_NAME_MAPPING[target_emotion_lower]
        elif target_emotion_lower in VALID_EMOTIONS_LOWER:
            true_label = VALID_EMOTIONS_LOWER[target_emotion_lower]
            
        if true_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_emotion)
        
        # Compile Row Data
        row_data = {
            "Text_Group": text_group,
            "Target_Emotion": target_emotion_raw,
            "Mapped_Label": true_label if true_label else "OOD (Ignored)",
            "Alpha": alpha_value,
            "Filename": filename,
            "Auto_Transcribed_Text": auto_text,
            "Predicted_Emotion": pred_emotion,
            "Confidence": f"{probs[pred_idx]*100:.2f}%"
        }
        for i, emo in IDX_TO_EMOTION.items():
            row_data[f"Prob_{emo}"] = round(probs[i], 4)
            
        results.append(row_data)

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Text_Group", "Target_Emotion", "Alpha"]) 
        output_csv = "synthetic_test_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n[System] Inference complete. Detailed results saved to '{output_csv}'.")

    # Generate Evaluation Report
    if y_true and y_pred:
        acc = accuracy_score(y_true, y_pred)
        print("\n" + "="*60)
        print(" MODEL EVALUATION REPORT (Supported Classes Only)")
        print("="*60)
        print(f" Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f" Valid Samples    : {len(y_true)} / {len(results)}")
        print("-" * 60)
        print(classification_report(y_true, y_pred, zero_division=0))
        print("="*60)
    else:
        print("\n[System Notice] No valid target emotions found for accuracy calculation.")
        print("Ensure folder names match standard classes or are defined in FOLDER_NAME_MAPPING.")

if __name__ == "__main__":
    main()