import os
import pandas as pd
import re

def extract_transcriptions(iemocap_dir):
    data = []
    # 遍歷 Session 1-5
    for session in range(1, 6):
        trans_dir = f"{iemocap_dir}/Session{session}/dialog/transcriptions"
        if not os.path.exists(trans_dir): continue
        
        for file in os.listdir(trans_dir):
            if file.endswith(".txt"):
                with open(os.path.join(trans_dir, file), "r") as f:
                    for line in f:
                        # 格式通常是: [時間區間] ID [情緒]: 文字
                        match = re.search(r'(\w+)\s\[.*\]:\s(.*)', line)
                        if match:
                            audio_id = match.group(1)
                            text = match.group(2).strip()
                            data.append({"audio_id": audio_id, "transcription": text})
    return pd.DataFrame(data)

# 讀取你原本的 8 分類 CSV 並合併文字
iemocap_path = "./IEMOCAP_Dataset" # 請確認路徑
original_csv = pd.read_csv("iemocap_8classes.csv")

print("正在提取逐字稿...")
df_trans = extract_transcriptions(iemocap_path)

# 透過音檔名稱進行合併 (假設 audio_name 包含 ID)
original_csv['audio_id'] = original_csv['audio_name'].apply(lambda x: os.path.basename(x).replace(".wav", ""))
final_df = pd.merge(original_csv, df_trans, on="audio_id", how="left")

# 儲存新 CSV
final_df.to_csv("iemocap_with_text.csv", index=False)
print("已生成 iemocap_with_text.csv！")