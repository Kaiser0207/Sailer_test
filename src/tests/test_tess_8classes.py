import os
import torch
import torch.nn as nn
import librosa
from transformers import WhisperProcessor, WhisperModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Definitions
# ==========================================
class WeightedAveragePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention_weights(x), dim=1)
        weighted_x = x * weights
        return weighted_x.sum(dim=1)

class SAILER_AcousticModel(nn.Module):
    def __init__(self, input_dim=1280, num_classes=8):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU()
        )
        self.attention_pooling = WeightedAveragePooling(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, acoustic_features):
        x = acoustic_features.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        x = self.attention_pooling(x)
        logits = self.classifier(x)
        return logits

# ==========================================
# Cross-Corpus Evaluation Pipeline
# ==========================================
if __name__ == "__main__":
    print(f"--- Initialization: TESS Cross-Corpus Evaluation ({device}) ---")

    dataset_dir = "toronto-emotional-speech-set-tess"

    # Load the trained 8-class checkpoint
    model = SAILER_AcousticModel(input_dim=1280, num_classes=8).to(device)
    model.load_state_dict(torch.load('best_sailer_8class.pth', map_location=device, weights_only=True))
    model.eval()

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device)

    # Label Mapping: Aligning TESS taxonomy to RAVDESS indexing
    # Note: TESS lacks a 'Calm' class. 'ps' stands for pleasant surprise.
    tess_to_ravdess_map = {
        'neutral': 0,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fear': 5,
        'disgust': 6,
        'ps': 7
    }

    # Inverse mapping for classification report generation
    ravdess_inverse_map = {
        0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
        4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
    }

    test_files = []
    
    # Traverse the local TESS directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                filename_no_ext = os.path.splitext(file)[0]
                parts = filename_no_ext.split('_')
                if len(parts) >= 3:
                    emotion_tag = parts[-1].lower()
                    if emotion_tag in tess_to_ravdess_map:
                        label_idx = tess_to_ravdess_map[emotion_tag]
                        test_files.append((os.path.join(root, file), label_idx))

    total_files = len(test_files)
    print(f"Total TESS samples aggregated for evaluation: {total_files}")

    all_predictions = []
    all_ground_truth = []

    # Inference execution
    with torch.no_grad():
        for audio_path, true_label in tqdm(test_files, desc="Cross-Corpus Inference"):
            audio_array, _ = librosa.load(audio_path, sr=16000)

            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)
            input_features = inputs.input_features.to(whisper_model.dtype)
            features = whisper_model.encoder(input_features).last_hidden_state

            speech_features = features.squeeze(0).float().cpu()
            logits = model(speech_features.unsqueeze(0).to(device))

            _, predicted_class = torch.max(logits, 1)

            all_predictions.append(predicted_class.cpu().item())
            all_ground_truth.append(true_label)

    print("\n--- Cross-Corpus Classification Report ---")
    target_names = [ravdess_inverse_map[i] for i in range(8)]
    
    # zero_division=0 handles the 'Calm' class which has 0 support in TESS
    print(classification_report(all_ground_truth, all_predictions, labels=list(range(8)), target_names=target_names, zero_division=0))

    accuracy = accuracy_score(all_ground_truth, all_predictions) * 100
    print(f"\nFinal TESS Cross-Corpus Accuracy: {accuracy:.2f}%")