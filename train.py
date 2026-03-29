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
    """
    SAILER 系統核心訓練迴圈 (Main Training Pipeline)。
    負責組裝巨型預訓練模型 (Whisper, RoBERTa)、載入特徵與標籤，執行多任務聯合訓練，並追蹤指標。
    """
    # ==========================================
    # 1. 訓練參數與基礎環境配置 (Configuration & Setup)
    # ==========================================
    config = {
        "model_name": "SAILER_BestSingleSystem",
        "epochs": 15,            
        "batch_size": 64,        
        "learning_rate": 0.0005, 
        "num_classes": 8,        
        "secondary_class_num": 17
    }

    tracker = ExperimentTracker(experiment_name=config["model_name"])
    tracker.save_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data" 

    # ==========================================
    # 2. 載入預估模型架構 (Load Pre-trained Encoders)
    # 策略: 載入 Whisper 與 RoBERTa，並將其神經網絡「完全凍結 (Frozen)」，專注訓練 SAILER 融合層
    # ==========================================
    r_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    # 載入 Whisper 編碼器，強制限制最大時間序列為 750 (轉換後為 15秒) 嚴格遵循論文規格
    whisper_enc = WhisperModel.from_pretrained(
        "openai/whisper-large-v3",
        max_source_positions=750,
        ignore_mismatched_sizes=True
    ).encoder.to(device)
    
    roberta_model = RobertaModel.from_pretrained("roberta-large").to(device)

    # 迴圈設定為 eval 模式，並停止計算計算圖梯度 (requires_grad = False) 以極大化釋放顯存
    for m in [whisper_enc, roberta_model]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # ==========================================
    # 3. 準備資料集加載器 (DataLoader Pipeline)
    # Train: 套用資料增強 (Audio Mixing) 與動態過採樣，增加資料多樣性保護少數群體
    # Validation: 純淨資料供給，嚴格依賴自然資料分佈作為評估基準
    # ==========================================
    train_dataset = MSP_Podcast_Dataset(data_dir, split="Train", roberta_tokenizer=r_tokenizer, apply_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    val_dataset = MSP_Podcast_Dataset(data_dir, split="Development", roberta_tokenizer=r_tokenizer, apply_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # ==========================================
    # 4. 初始化自定義 SAILER 模型與訓練組件 (Model Initialization)
    # ==========================================
    model = SAILER_Model(
        num_classes=config["num_classes"],
        secondary_class_num=config["secondary_class_num"],
        dropout_rate=0.2
    ).to(device)

    # 引入 torch.compile，加速訓練速度
    try:
        print("正在啟動編譯優化 (torch.compile)...")
        model = torch.compile(model)
        print("編譯成功！")
    except Exception as e:
        print(f"編譯失敗，將使用一般動態圖模式訓練。錯誤: {e}")

    # 定義多任務損失函數 (Multi-task Loss Functions)
    # 針對軟標籤(Soft labels)分佈預測使用 KL Divergence，針對連續值則採用均方差 MSE
    criterion_primary = nn.KLDivLoss(reduction='batchmean')
    criterion_secondary = nn.KLDivLoss(reduction='batchmean')
    criterion_avd = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    best_min_map = 0.0

    # ==========================================
    # 5. 主訓練迴圈 (Main Epoch Execution)
    # ==========================================
    for epoch in range(config["epochs"]):
        print(f"\n[{epoch+1}/{config['epochs']}] Epoch 訓練開始...")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            # 解離並裝載這包 Batch 產生的 7 個輸入張量至 GPU
            w_feat, t_ids, t_mask, label_dists, sec_dists, avd_targets, lengths = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # ==== 階段一：預訓練前端提取 (No Grad Context) ====
                with torch.no_grad():
                    w_seq = whisper_enc(w_feat).last_hidden_state  
                    roberta_out = roberta_model(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                    t_hidden_states = roberta_out.hidden_states  
                
                # ==== 階段二：SAILER 特徵融合網絡 (Train Context) ====
                primary_logits, secondary_logits, arousal, valence, dominance = model(
                    w_seq, t_hidden_states, t_mask, lengths
                )

                # 計算 5 大任務的 Loss 差異
                loss_primary = criterion_primary(F.log_softmax(primary_logits, dim=-1), label_dists)
                loss_secondary = criterion_secondary(F.log_softmax(secondary_logits, dim=-1), sec_dists)
                avd_pred = torch.cat([arousal, valence, dominance], dim=-1) 
                loss_avd = criterion_avd(avd_pred, avd_targets)
                
                # Unweighted Sum 無權重總和
                loss = loss_primary + loss_secondary + loss_avd

            # ==== 階段三：傳播梯度更新 ====
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            wandb.log({
                "batch_loss": loss.item(),
                "batch_primary_loss": loss_primary.item(),
                "batch_secondary_loss": loss_secondary.item(),
                "batch_avd_loss": loss_avd.item(),
            })

        avg_train_loss = total_train_loss / len(train_loader)

        # ==========================================
        # 6. 階段性驗證循環 (Validation Inference & Metrics)
        # 用於衡量模型在未見過資料 (Dev Set) 中的表現，嚴防Overfitting
        # ==========================================
        model.eval()
        total_val_loss = 0
        all_preds, all_labels, all_preds_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                w_feat, t_ids, t_mask, label_dists, sec_dists, avd_targets, lengths = [b.to(device) for b in batch]

                with torch.amp.autocast('cuda'):
                    w_seq = whisper_enc(w_feat).last_hidden_state
                    roberta_out = roberta_model(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                    t_hidden_states = roberta_out.hidden_states

                    primary_logits, secondary_logits, arousal, valence, dominance = model(
                        w_seq, t_hidden_states, t_mask, lengths
                    )
                    
                    val_loss_primary = criterion_primary(F.log_softmax(primary_logits, dim=-1), label_dists)
                    val_loss_secondary = criterion_secondary(F.log_softmax(secondary_logits, dim=-1), sec_dists)
                    avd_pred = torch.cat([arousal, valence, dominance], dim=-1)
                    val_loss_avd = criterion_avd(avd_pred, avd_targets)
                    val_loss = val_loss_primary + val_loss_secondary + val_loss_avd
                    total_val_loss += val_loss.item()

                probs = F.softmax(primary_logits, dim=-1)
                _, predicted = torch.max(primary_logits, 1)
                _, true_labels = torch.max(label_dists, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
                all_preds_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # 驗證整體精準指標
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

        # 針對極端少數類別單獨計算平均精度 (Average Precision, AP)。
        # 在高度失衡的資料集中，比起單看 F1 這個指標能在微觀保護力上給予更清晰的反饋。
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

        # ==========================================
        # 7. 雙重保險權重備份 (Checkpoint Check&Save)
        # ==========================================
        
        if min_map > best_min_map:
            best_min_map = min_map
            print(f"少數類別準確度 (Min. mAP): {min_map:.4f}，儲存特化權重...")
            torch.save(model.state_dict(), os.path.join(tracker.weights_dir, "best_model_min_map.pth"))

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            print(f"發現更高 (Macro F1): {val_macro_f1:.4f}，儲存最強綜合權重...")
            torch.save(model.state_dict(), os.path.join(tracker.weights_dir, "best_model_f1.pth"))

    tracker.close()

if __name__ == "__main__":
    main()
