import torch
import gc
from transformers import RobertaModel, WhisperModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sailer_model import SAILER_Model

def print_vram(step_name):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{step_name:<25}] 實際佔用 (Allocated): {allocated:.2f} GB | PyTorch 保留 (Reserved): {reserved:.2f} GB")

def main():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    
    print("====== 測試載入『完整版 Whisper-Large-V3 (含解碼器)』======\n")
    print_vram("1. 初始空載狀態")
    
    # 改成載入完整 Whisper，拿掉 .encoder
    whisper_full = WhisperModel.from_pretrained(
        "openai/whisper-large-v3"
    ).to(device)
    print_vram("2. 載入完整 Whisper")
    
    roberta_model = RobertaModel.from_pretrained("roberta-large").to(device)
    print_vram("3. 載入 RoBERTa-Large")
    
    model = SAILER_Model(num_classes=8, secondary_class_num=17, dropout_rate=0.2).to(device)
    print_vram("4. 載入 SAILER 融合層")
    
    for m in [whisper_full, roberta_model]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
            
    print_vram("5. 模型權重凍結 (Freeze)")
    print("\n==============================================")

if __name__ == "__main__":
    main()
