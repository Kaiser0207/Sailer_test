import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, WhisperModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# 載入你寫好的模組
from code.msp_dataset import MSP_Podcast_Dataset
from code.sailer_model import SAILER_Model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # ==========================================
    # 1. 路徑設定 (請確認這裡指向你的最佳模型)
    # ==========================================
    data_dir = "/home/brant/Project/SAILER_test/MSP_Podcast_Data"
    # 👉 替換成你實際的實驗資料夾名稱
    experiment_dir = "experiments/20260325_012852_SAILER_IS25_Final_15s" 
    model_path = os.path.join(experiment_dir, "weights", "best_model.pth")

    # ==========================================
    # 2. 載入預訓練特徵提取器
    # ==========================================
    print("載入基礎模型 (Whisper & RoBERTa)...")
    r_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    whisper_enc = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.to(device)
    whisper_enc.eval()
    
    roberta_model = RobertaModel.from_pretrained("roberta-large").to(device)
    roberta_model.eval()

    # ==========================================
    # 3. 載入 SAILER 最強大腦權重
    # ==========================================
    print("載入最佳 SAILER 權重...")
    model = SAILER_Model(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ==========================================
    # 4. 準備驗證資料集
    # ==========================================
    val_dataset = MSP_Podcast_Dataset(data_dir, split="Development", 
                                      roberta_tokenizer=r_tokenizer, apply_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=6)

    # ==========================================
    # 5. 進行全面推論
    # ==========================================
    all_preds = []
    all_labels = []

    print("開始測試驗證集...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing"):
            w_feat, t_ids, t_mask, label_dists = [b.to(device) for b in batch]
            
            # 補零並轉型 (對齊你訓練時的做法)
            w_feat_padded = F.pad(w_feat, (0, 1500))
            w_seq = whisper_enc(w_feat_padded.to(whisper_enc.dtype)).last_hidden_state.float()
            t_seq = roberta_model(input_ids=t_ids, attention_mask=t_mask).last_hidden_state.float()

            logits = model(w_seq, t_seq, t_mask)
            
            # 取最大機率作為預測結果
            _, predicted = torch.max(logits, 1)
            _, true_labels = torch.max(label_dists, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    # ==========================================
    # 6. 繪製超精美 Normalized Confusion Matrix
    # ==========================================
    # 對應 emotion_map: {'N':0, 'A':1, 'S':2, 'H':3, 'F':4, 'D':5, 'U':6, 'C':7}
    class_names = ['Neutral', 'Angry', 'Sad', 'Happy', 'Fear', 'Disgust', 'Surprise', 'Contempt']
    
    # 計算混淆矩陣，並按「真實標籤 (true)」進行百分比正規化
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    plt.figure(figsize=(10, 8))
    # 使用 seaborn 畫熱力圖，cmap="Blues" 是學術界最愛的顏色
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12}) # 數字字體大小

    plt.title('SAILER Emotion Recognition - Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Emotion', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 儲存高解析度圖片
    save_fig_path = os.path.join(experiment_dir, "confusion_matrix_HD.png")
    plt.savefig(save_fig_path, dpi=300)
    print(f"圖表已儲存至: {save_fig_path}")

if __name__ == "__main__":
    main()