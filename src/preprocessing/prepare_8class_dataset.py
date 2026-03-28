import os
import pandas as pd

def create_full_ravdess_csv(raw_data_dir, output_csv):
    """
    Parses RAVDESS audio filenames and generates a complete CSV index 
    for 8-class emotion classification.
    """
    emotion_mapping = {
        '01': 'Neutral',
        '02': 'Calm',
        '03': 'Happy',
        '04': 'Sad',
        '05': 'Angry',
        '06': 'Fearful',
        '07': 'Disgust',
        '08': 'Surprised'
    }

    dataset = []

    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.wav'):
                identifiers = file.split('.')[0].split('-')

                if len(identifiers) == 7:
                    emotion_code = identifiers[2]
                    
                    if emotion_code in emotion_mapping:
                        emotion_label = emotion_mapping[emotion_code]
                        full_path = os.path.join(root, file)

                        dataset.append({
                            'audio_name': full_path,
                            'primary_emotion': emotion_label
                        })

    df = pd.DataFrame(dataset)
    
    # Check if dataset is empty to prevent KeyError
    if len(df) == 0:
        print(f"Error: 找不到任何符合格式的音檔！請確認路徑 '{raw_data_dir}' 是否正確。")
        return

    df.to_csv(output_csv, index=False)
    
    print(f"Dataset generated successfully: {output_csv}")
    print(f"Total samples collected: {len(df)}")
    print("-" * 40)
    print("Class distribution:")
    print(df['primary_emotion'].value_counts().to_string())

if __name__ == "__main__":
    # Updated to match the folder name in your VS Code explorer
    RAVDESS_DIR = "./RAVDESS_Dataset" 
    OUTPUT_CSV = "my_dataset.csv"

    create_full_ravdess_csv(RAVDESS_DIR, OUTPUT_CSV)