import torch
import torch.nn as nn
from train_4class import collate_fn
from src.interview_dataset import InterviewEmotionDataset
from src.model.emotion.whisper_emotion import WhisperWrapper
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset
    val_dataset = InterviewEmotionDataset("datasets/MSP_Podcast_Data", split="Development")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 2. Load Model & Pretrained Head
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
    hidden_dim = 256
    num_classes = 4
    model.emotion_layer = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)
    
    head_path = "experiments/20260418_234018_SAILER_v4.0_Interview_4Class/weights/best_emotion_head.pth"
    print(f"Loading weights from: {head_path}")
    model.emotion_layer.load_state_dict(torch.load(head_path, map_location=device))
    
    # 3. Evaluate Immediately
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n=== 即時驗證報告 (零訓練) ===")
    print(classification_report(all_labels, all_preds, target_names=['Neutral', 'Happy', 'Fear', 'Surprise']))

if __name__ == "__main__":
    test()
