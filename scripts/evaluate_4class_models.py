import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import wandb

# 引用我們客製化的 4-Class Dataset 與模型結構
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "vox-profile-release"))
from src.interview_dataset import InterviewEmotionDataset
from src.model.emotion.whisper_emotion import WhisperWrapper

def collate_fn(batch):
    waveforms, labels, lengths = zip(*batch)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return list(waveforms), labels, lengths

def load_data(split="Test1"):
    data_dir = "datasets/MSP_Podcast_Data"
    dataset = InterviewEmotionDataset(data_dir, split=split)
    # 將 batch 設為 16 或 32，確保記憶體安全
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return dataset, dataloader

def evaluate_model(name, model, dataloader, device, split, baseline_mode=False):
    print(f"\n🚀 開始評估模型: {name}")
    model.eval()
    all_preds = []
    all_labels = []

    # 官方模型 (tiantiaf/whisper-large-v3-msp-podcast-emotion) 的 9 個輸出的順序為:
    # 0:Anger, 1:Contempt, 2:Disgust, 3:Fear, 4:Happiness, 5:Neutral, 6:Sadness, 7:Surprise, 8:Other
    # 我們要對應的 4 類 (Neutral, Happy, Fear, Surprise) 在官方模型中的索引為:
    target_indices_baseline = [5, 4, 3, 7]

    with torch.no_grad():
        for waveforms, labels, lengths in tqdm(dataloader, desc=f"Evaluating {name}"):
            labels = labels.cpu().numpy()
            lengths = lengths.to(device)
            # 官方模型接受 waveforms_list 作為輸入
            with torch.amp.autocast('cuda'):
                # WhisperWrapper forward(return_feature=False)
                outputs = model(waveforms, length=lengths)
                logits = outputs[0]

            if baseline_mode:
                # 抽取 4 個目標類別的 logits：5(N), 4(H), 3(F), 7(U)
                sub_logits = logits[:, target_indices_baseline]
                preds = torch.argmax(sub_logits, dim=-1).cpu().numpy()
            else:
                # Phase 1~3 訓練時就是用 0,1,2,3
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    target_names = ['Neutral', 'Happy', 'Fear', 'Surprise']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(f"\n{'='*50}\n 📊 驗證報告 ({name} - {split})\n{'-'*50}\n{report}{'='*50}")

    # 計算 Wandb metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cls_f1 = f1_score(all_labels, all_preds, average=None)

    metrics = {
        f"{split}/{name}_Accuracy": acc,
        f"{split}/{name}_Macro_F1": macro_f1,
        f"{split}/{name}_Neutral_F1": cls_f1[0],
        f"{split}/{name}_Happy_F1": cls_f1[1],
        f"{split}/{name}_Fear_F1": cls_f1[2],
        f"{split}/{name}_Surprise_F1": cls_f1[3],
    }
    return metrics

def evaluate_all_for_split(split, device):
    print(f"\n" + "="*80)
    print(f" 🎯 開始評估資料切分: {split} ")
    print(f"="*80)

    dataset, dataloader = load_data(split=split)
    metrics = {}

    # ===== 1. Baseline 模型 (官方 8-Class) =====
    print("\n[載入] tiantiaf/whisper-large-v3-msp-podcast-emotion (Baseline)")
    baseline_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion", 
        output_class_num=9
    ).float().to(device)
    m1 = evaluate_model("Baseline", baseline_model, dataloader, device, split, baseline_mode=True)
    metrics.update(m1)
    del baseline_model
    torch.cuda.empty_cache()

    # ===== 2. Phase 1 模型 =====
    print("\n[載入] Phase 1 (Best Epoch 4)")
    # 技巧：先以 9 類載入官方權重避免 from_pretrained 報錯，再手動換成 4 類
    phase1_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion", 
        output_class_num=9
    ).float().to(device)
    # 手動更換為 4 類分類頭 (必須與 train_4class.py 結構完全一致：含 Dropout)
    import torch.nn as nn
    phase1_model.emotion_layer = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2), # 必須為 4 層結構以匹配 state_dict (0,1,2,3)
        nn.Linear(256, 4),
    ).to(device)
    
    phase1_ckpt = "experiments/20260418_234018_SAILER_v4.0_Interview_4Class/weights/best_emotion_head.pth"
    if os.path.exists(phase1_ckpt):
        phase1_model.emotion_layer.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        m2 = evaluate_model("Phase_1", phase1_model, dataloader, device, split)
        metrics.update(m2)
    del phase1_model
    torch.cuda.empty_cache()

    # ===== 3. Phase 2 模型 =====
    print("\n[載入] Phase 2 (Epoch 30 Latest Checkpoint)")
    phase2_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion", 
        output_class_num=9
    ).float().to(device)
    phase2_model.emotion_layer = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4),
    ).to(device)

    # 路徑：20260423_020544 是我們剛跑出來的 0.5445 版本
    p2_dir = "experiments/20260423_020544_SAILER_v4.0_Phase2_FinalPeak_NoSampler/weights"
    if os.path.exists(p2_dir):
        # 載入分類頭
        phase2_model.emotion_layer.load_state_dict(torch.load(os.path.join(p2_dir, "best_emotion_head.pth"), map_location=device))
        # 載入中間層 (這是 Phase 2 的核心)
        phase2_model.model_seq.load_state_dict(torch.load(os.path.join(p2_dir, "best_model_seq.pth"), map_location=device))
        m3 = evaluate_model("Phase_2", phase2_model, dataloader, device, split)
        metrics.update(m3)
    del phase2_model
    torch.cuda.empty_cache()

    # ===== 4. Phase 3 模型 =====
    print("\n[載入] Phase 3 (Best Epoch 21)")
    phase3_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion", 
        output_class_num=9
    ).float().to(device)
    phase3_model.emotion_layer = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4),
    ).to(device)

    phase3_dir = "experiments/20260419_212802_SAILER_v4.0_Interview_4Class/weights"
    if os.path.exists(os.path.join(phase3_dir, "best_emotion_head.pth")):
        phase3_model.emotion_layer.load_state_dict(torch.load(os.path.join(phase3_dir, "best_emotion_head.pth"), map_location=device))
        phase3_model.model_seq.load_state_dict(torch.load(os.path.join(phase3_dir, "best_model_seq.pth"), map_location=device))
        phase3_model.backbone_model.encoder.layers[-1].load_state_dict(torch.load(os.path.join(phase3_dir, "best_whisper_last_layer.pth"), map_location=device))
        m4 = evaluate_model("Phase_3", phase3_model, dataloader, device, split)
        metrics.update(m4)
    del phase3_model
    torch.cuda.empty_cache()

    # 傳送到 W&B 本 split 的總結
    wandb.log(metrics)

def main():
    wandb.init(
        project="SAILER_Emotion_Recognition",
        name="V4.0_4Class_Benchmark",
        tags=["benchmark", "interview"],
        notes="評估 Baseline, Phase 1, Phase 2, Phase 3 在 Test1 和 Test2 上的表現"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用硬體設備: {device}")

    for split in ["Test1", "Test2"]:
        evaluate_all_for_split(split, device)

    wandb.finish()

if __name__ == "__main__":
    main()
