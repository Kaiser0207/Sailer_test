import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split # 引入 sklearn 的資料切割神器

def create_and_split_iemocap_csv(raw_data_dir, output_train_csv, output_test_csv):
    """
    Parses IEMOCAP evaluation text files, generates a complete dataset,
    and performs a stratified 80/20 train/test split.
    """
    emotion_mapping = {
        'neu': 'Neutral',
        'fru': 'Frustrated', 
        'ang': 'Angry',
        'sad': 'Sad',
        'exc': 'Excited',    
        'hap': 'Happy',
        'sur': 'Surprised',
        'fea': 'Fearful'
    }

    dataset = []
    pattern = re.compile(r'^\[.+\]\s+(Ses.+)\s+([a-z]{3}|xxx)\s+\[.+\]')

    print(f"開始解析 IEMOCAP 資料夾: {raw_data_dir} ...")

    # 1. 讀取並解析所有資料
    for session in range(1, 6):
        eval_dir = os.path.join(raw_data_dir, f"Session{session}", "dialog", "EmoEvaluation")
        
        if not os.path.exists(eval_dir):
            continue

        for txt_file in os.listdir(eval_dir):
            if not txt_file.endswith('.txt'):
                continue

            with open(os.path.join(eval_dir, txt_file), 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        turn_name = match.group(1)
                        emotion_code = match.group(2)

                        if emotion_code in emotion_mapping:
                            emotion_label = emotion_mapping[emotion_code]
                            folder_name = "_".join(turn_name.split("_")[:-1])
                            full_path = os.path.join(raw_data_dir, f"Session{session}", "sentences", "wav", folder_name, f"{turn_name}.wav")

                            if os.path.exists(full_path):
                                dataset.append({
                                    'audio_name': full_path,
                                    'primary_emotion': emotion_label
                                })

    df = pd.DataFrame(dataset)
    
    if len(df) == 0:
        print(f"Error: 找不到任何符合格式的音檔！請確認路徑 '{raw_data_dir}' 是否正確。")
        return

    print(f"✅ 解析完成！共抓取 {len(df)} 筆有效音檔。")
    print("準備進行 80/20 分層切割 (Stratified Split)...")

    # 2. 執行 80/20 分層切割
    # stratify=df['primary_emotion'] 是靈魂所在，確保極端少數類別(Fearful)也能被均勻分配
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, # 設定隨機種子，確保每次切割結果一樣，方便重現實驗
        stratify=df['primary_emotion'] 
    )

    # 3. 輸出兩個 CSV 檔案
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)
    
    # 4. 印出超專業的數據統計報告
    print("-" * 50)
    print(f"🎉 訓練集 (Train) 儲存至: {output_train_csv} (共 {len(train_df)} 筆)")
    print(train_df['primary_emotion'].value_counts().to_string())
    print("-" * 50)
    print(f"🎯 測試集 (Test)  儲存至: {output_test_csv} (共 {len(test_df)} 筆)")
    print(test_df['primary_emotion'].value_counts().to_string())
    print("-" * 50)

if __name__ == "__main__":
    IEMOCAP_DIR = "./IEMOCAP_Dataset" 
    TRAIN_CSV = "train_iemocap.csv"
    TEST_CSV = "test_iemocap.csv"

    create_and_split_iemocap_csv(IEMOCAP_DIR, TRAIN_CSV, TEST_CSV)