import pandas as pd
import wave
import os
from tqdm import tqdm

csv_path = "/home/brant/Project/SAILER_test/Crab/data/msp2_processed_labels.csv"
audio_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"

print(f"Reading CSV: {csv_path}")
df = pd.read_csv(csv_path)
filenames = df['FileName'].tolist()

over_14 = []
max_dur = 0
max_file = ""

print("Scanning audio durations (this might take a minute)...")
for f in tqdm(filenames):
    path = os.path.join(audio_dir, f)
    if not os.path.exists(path):
        continue
    
    try:
        with wave.open(path, 'r') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            
            if duration > 14:
                over_14.append((f, duration))
            
            if duration > max_dur:
                max_dur = duration
                max_file = f
    except Exception as e:
        pass

print("\n" + "="*50)
print(f"掃描完成！總共檢查了 {len(filenames)} 個檔案")
print(f"最長的音檔: {max_file} ({max_dur:.2f} 秒)")
print(f"超過 14 秒的音檔數量: {len(over_14)}")
if len(over_14) > 0:
    print("\n前 5 個超長檔案:")
    for f, d in over_14[:5]:
        print(f"  - {f}: {d:.2f} 秒")
print("="*50)
