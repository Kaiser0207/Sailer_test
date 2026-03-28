from transformers import WhisperProcessor, RobertaTokenizer
from code.msp_dataset import MSP_Podcast_Dataset
from torch.utils.data import DataLoader

print("1. 載入模型處理器 (Processors)...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

print("\n2. 初始化 MSP-Podcast Validation 資料集...")
data_dir = "/home/brant/Project/SAILER_test/MSP_Podcast_Data"
val_dataset = MSP_Podcast_Dataset(
    data_dir=data_dir, 
    split="Development",  
    whisper_processor=whisper_processor, 
    roberta_tokenizer=roberta_tokenizer
)

print("\n3. 測試 DataLoader 取出第一批資料 (Batch Size = 4)...")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# 只拿第一個 Batch 來看看形狀對不對
w_feat, t_ids, t_mask, labels = next(iter(val_loader))

print("======================================")
print(f"Whisper 特徵形狀: {w_feat.shape} (應該要是 [4, 128, 3000])")
print(f"RoBERTa ID 形狀: {t_ids.shape} (應該要是 [4, 128])")
print(f"RoBERTa Mask 形狀: {t_mask.shape} (應該要是 [4, 128])")
print(f"真實標籤: {labels} (應該要是 4 個 0~7 的數字)")
print("======================================")