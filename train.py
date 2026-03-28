import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, WhisperModel
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, classification_report
import numpy as np
import os
import wandb

from src.experiment_tracker import ExperimentTracker
from src.msp_dataset import MSP_Podcast_Dataset
from src.sailer_model import SAILER_Model

def main():
    config = {
        "model_name": "SAILER_IS25_Full_WeightedAvg",
        "epochs": 15,
        "batch_size": 64,
        "learning_rate": 0.0004,
        "num_classes": 8
    }

    wandb.init(project="SAILER_Emotion_Recognition", name=config["model_name"], config=config)
    tracker = ExperimentTracker(experiment_name=config["model_name"])
    tracker.save_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data" 

    r_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    whisper_enc = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.to(device)
    roberta_model = RobertaModel.from_pretrained("roberta-large").to(device)

    for m in [whisper_enc, roberta_model]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    train_dataset = MSP_Podcast_Dataset(data_dir, split="Train", roberta_tokenizer=r_tokenizer, apply_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    val_dataset = MSP_Podcast_Dataset(data_dir, split="Development", roberta_tokenizer=r_tokenizer, apply_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = SAILER_Model(num_classes=config["num_classes"],dropout_rate=0.2).to(device)

    try:
        print("正在啟動編譯優化 (torch.compile)...")
        model = torch.compile(model)
        print("編譯成功！")
    except Exception as e:
        print(f"編譯失敗，將使用原始模式訓練。錯誤: {e}")

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    best_min_map = 0.0

    for epoch in range(config["epochs"]):
        print(f"\n[{epoch+1}/{config['epochs']}] Epoch 訓練開始...")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            w_feat, t_ids, t_mask, label_dists = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    w_seq = whisper_enc(w_feat).last_hidden_state  
                    t_seq = roberta_model(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
                logits = model(w_seq, t_seq, t_mask) 
                loss = criterion(F.log_softmax(logits, dim=-1), label_dists)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 驗證 ---
        model.eval()
        total_val_loss = 0
        all_preds, all_labels, all_preds_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                w_feat, t_ids, t_mask, label_dists = [b.to(device) for b in batch]

                with torch.amp.autocast('cuda'):
                    w_seq = whisper_enc(w_feat).last_hidden_state
                    t_seq = roberta_model(input_ids=t_ids, attention_mask=t_mask).last_hidden_state

                    logits = model(w_seq, t_seq, t_mask)
                    val_loss = criterion(F.log_softmax(logits, dim=-1), label_dists)
                    total_val_loss += val_loss.item()

                probs = F.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                _, true_labels = torch.max(label_dists, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
                all_preds_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        emotion_names = ['Neutral', 'Angry', 'Sad', 'Happy', 'Fear', 'Disgust', 'Surprise', 'Contempt']
        minority_classes = [4, 5, 6, 7]
        minority_labels = {4: "Fear", 5: "Disgust", 6: "Surprise", 7: "Contempt"}
        min_aps = []
        
        log_dict = {
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss
        }

        for c in minority_classes:
            y_true_c = (np.array(all_labels) == c).astype(int)
            y_score_c = np.array(all_preds_probs)[:, c]
            if np.sum(y_true_c) > 0:
                ap = average_precision_score(y_true_c, y_score_c)
                min_aps.append(ap)
                log_dict[f"val_AP_{minority_labels[c]}"] = ap
                
        min_map = np.mean(min_aps) if min_aps else 0.0
        log_dict["val_min_mAP"] = min_map

        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1} 驗證結果")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Min. mAP: {min_map:.4f}")
        print(f"{'-'*40}")
        print(classification_report(all_labels, all_preds, labels=list(range(8)), target_names=emotion_names, zero_division=0))
        print(f"{'='*40}\n")

        tracker.log_metrics(epoch, avg_train_loss, avg_val_loss, val_acc)
        wandb.log(log_dict, step=epoch)

        if min_map > best_min_map:
            best_min_map = min_map
            print(f"少數類別預測 (Min. mAP): {min_map:.4f}，儲存權重...")
            torch.save(model.state_dict(), os.path.join(tracker.weights_dir, "best_model_min_map.pth"))

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            print(f"發現更高 F1: {val_macro_f1:.4f}，儲存權重...")
            torch.save(model.state_dict(), os.path.join(tracker.weights_dir, "best_model_2.pth"))

    tracker.close()
    wandb.finish()

if __name__ == "__main__":
    main()