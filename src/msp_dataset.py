import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

class MSP_Podcast_Dataset(Dataset):
    """
    MSP-Podcast 語音情感資料集裝載器 (PyTorch Dataset)。
    負責讀取、對齊特徵 (15s Whisper 聲紋與 128-token RoBERTa 文本)，
    並在 Train 階段執行資料增強 (Audio Mixing) 與動態標籤平滑 (Annotation Dropout)。
    """
    def __init__(self, data_dir, split="Train", roberta_tokenizer=None, apply_aug=True):
        self.data_dir = data_dir
        self.split = split
        self.roberta_tokenizer = roberta_tokenizer
        self.apply_aug = apply_aug and split == "Train"
        
        self.feat_dir = os.path.join(data_dir, "Whisper_Features_15s")
        self.transcripts_dir = os.path.join(data_dir, "Transcripts")
        
        self.consensus_path = os.path.join(data_dir, "Labels", "labels_consensus.csv")
        self.detailed_path = os.path.join(data_dir, "Labels", "labels_detailed.csv")
        
        # 8 類核心情緒 (Primary)
        self.emotion_map_short = {'N': 0, 'A': 1, 'S': 2, 'H': 3, 'F': 4, 'D': 5, 'U': 6, 'C': 7}
        self.emotion_map_long = {'Neutral': 0, 'Angry': 1, 'Sad': 2, 'Happy': 3, 'Fear': 4, 'Disgust': 5, 'Surprise': 6, 'Contempt': 7}
        
        # 17 類 Secondary Emotion (含 8 核心 + 9 個 Other 子類)
        self.secondary_emotion_map = {
            'Neutral': 0, 'Angry': 1, 'Sad': 2, 'Happy': 3, 
            'Fear': 4, 'Disgust': 5, 'Surprise': 6, 'Contempt': 7,
            'Other-Concerned': 8, 'Other-Annoyed': 9, 'Other-Frustrated': 10,
            'Other-Confused': 11, 'Other-Amused': 12, 'Other-Disappointed': 13,
            'Other-Excited': 14, 'Other-Bored': 15, 'Other': 16
        }
        self.num_secondary_classes = 17
        
        self.majority_classes = [0, 1, 2, 3]
        self.minority_classes = [4, 5, 6, 7]
        
        print("從 labels_detailed.csv 聚合真實得票分佈...")
        self.vote_dict, self.secondary_vote_dict = self._build_vote_dictionary()
        
        self.data_records = self._load_data()

        class_counts = np.zeros(8)
        for r in self.data_records:
            class_counts[r['consensus_label']] += 1
        q = class_counts / class_counts.sum()
        w = 1.0 / (q + 1e-8)
        self.w_norm = w / w.sum()

        self.minority_records = [r for r in self.data_records if r['consensus_label'] in self.minority_classes]
        minority_counts = np.array([
            sum(1 for r in self.data_records if r['consensus_label'] == c)
            for c in self.minority_classes
        ])
        inv_weights = 1.0 / (minority_counts + 1e-8)
        class_weights = (inv_weights / inv_weights.sum()).tolist()

        weight_dict = {}
        for c, cw, count in zip(self.minority_classes, class_weights, minority_counts):
            weight_dict[c] = cw / (count + 1e-8)

        self.record_weights = [weight_dict[r['consensus_label']] for r in self.minority_records]    

        print(f"成功載入 [{self.split}] 特徵資料集，共 {len(self.data_records)} 筆！")

    def _build_vote_dictionary(self):
        """
        從詳盡標註檔 (labels_detailed.csv) 聚合原始標註員投票。
        SAILER 論文指出：比起使用絕對的 Hard Label (單一情緒)，
        讓模型學習人類間產生分歧的「投票機率分佈 (Soft Labels / Vote Distribution)」能帶來極大的抗雜訊效果。
        """
        df_detail = pd.read_csv(self.detailed_path)
        
        # === Primary emotion votes (8 classes) ===
        df_primary = df_detail[df_detail['EmoClass_Major'].isin(self.emotion_map_long.keys())]
        vote_dict = {}
        grouped = df_primary.groupby(['FileName', 'EmoClass_Major']).size().unstack(fill_value=0)
        for filename, row in grouped.iterrows():
            votes = np.zeros(8, dtype=np.float32)
            for emo_str, count in row.items():
                if emo_str in self.emotion_map_long:
                    idx = self.emotion_map_long[emo_str]
                    votes[idx] = count
            vote_dict[filename] = votes
        
        # === Secondary emotion votes (17 classes) ===
        secondary_vote_dict = {}
        def map_to_17(emo_str):
            if emo_str in self.secondary_emotion_map:
                return self.secondary_emotion_map[emo_str]
            # 處理各種 Other
            emo_lower = emo_str.lower() if isinstance(emo_str, str) else ""
            for key in ['concerned', 'annoyed', 'frustrated', 'confused', 
                       'amused', 'disappointed', 'excited', 'bored']:
                if key in emo_lower:
                    return self.secondary_emotion_map[f'Other-{key.capitalize()}']
            # 其餘所有 Other-* 歸為 catch-all
            if 'other' in emo_lower:
                return 16  
            return None
        
        for filename, grp in df_detail.groupby('FileName'):
            votes_17 = np.zeros(self.num_secondary_classes, dtype=np.float32)
            for emo_str in grp['EmoClass_Major']:
                idx = map_to_17(emo_str)
                if idx is not None:
                    votes_17[idx] += 1
            if votes_17.sum() > 0:
                secondary_vote_dict[filename] = votes_17
        
        return vote_dict, secondary_vote_dict

    def _load_data(self):
        """
        過濾並載入資料集索引表。
        策略亮點：在 Train 階段特意保留了 "No-Agreement" (無法達成共識) 的模糊樣本，
        這大幅擴充了模型的訓練池，並讓模型學會在混亂語境中抓取 Secondary Emotion。
        """
        records = []
        df_consensus = pd.read_csv(self.consensus_path)

        if self.split == "Train":
            # 1. 抓取所有純 Train Set 樣本 (包含共識與不共識)
            df_train = df_consensus[df_consensus['Split_Set'] == "Train"]
            
            # 2. 論文隱藏細節：抓取 Dev Set 中廢棄的 (No agreement / Other) 樣本，偷渡進 Train Set 擴充資料量
            df_dev = df_consensus[df_consensus['Split_Set'] == "Development"]
            df_dev_unagreed = df_dev[~df_dev['EmoClass'].isin(self.emotion_map_short.keys())]
            
            # 兩者合併成為最龐大、混合度最高的訓練前線池
            df_use = pd.concat([df_train, df_dev_unagreed])
        else:
            df_use = df_consensus[df_consensus['Split_Set'] == self.split]
            df_use = df_use[df_use['EmoClass'].isin(self.emotion_map_short.keys())]

        for _, row in tqdm(df_use.iterrows(), total=len(df_use), desc=f"Scanning {self.split}"):
            filename = row['FileName']
            feat_path = os.path.join(self.feat_dir, filename.replace('.wav', '.pt'))
            txt_path = os.path.join(self.transcripts_dir, filename.replace('.wav', '.txt'))
            
            if not os.path.exists(feat_path) or not os.path.exists(txt_path):
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Primary votes
            votes = self.vote_dict.get(filename, None)
            if votes is None or votes.sum() == 0:
                emo_key = row.get('EmoClass', None)
                votes = np.zeros(8, dtype=np.float32)
                if emo_key in self.emotion_map_short:
                    votes[self.emotion_map_short[emo_key]] = 1.0
                else:
                    votes = np.ones(8, dtype=np.float32) / 8.0  

            # Secondary votes (17 classes)
            sec_votes = self.secondary_vote_dict.get(filename, None)
            if sec_votes is None or sec_votes.sum() == 0:
                sec_votes = np.zeros(self.num_secondary_classes, dtype=np.float32)
                emo_key = row.get('EmoClass', None)
                if emo_key in self.emotion_map_short:
                    sec_votes[self.emotion_map_short[emo_key]] = 1.0 
                else:
                    sec_votes = np.ones(self.num_secondary_classes, dtype=np.float32) / self.num_secondary_classes

            arousal = (float(row.get('EmoAct', 4.0)) - 1.0) / 6.0
            valence = (float(row.get('EmoVal', 4.0)) - 1.0) / 6.0
            dominance = (float(row.get('EmoDom', 4.0)) - 1.0) / 6.0
            avd = np.array([arousal, valence, dominance], dtype=np.float32)

            consensus_label = self.emotion_map_short.get(row.get('EmoClass', None), int(np.argmax(votes)))

            records.append({
                "feat_path": feat_path,
                "text": text,
                "consensus_label": consensus_label,
                "votes": votes,
                "secondary_votes": sec_votes,
                "avd": avd,
            })
        return records

    def _get_target_distribution(self, votes, is_training):
        """
        計算主要情緒 (Primary Emotion) 的標籤分佈。
        防禦性機制: 包含 Annotation Dropout (丟棄部分強勢情緒投票) 與 
        Class Re-weighting (少數類別權重放大機制)，以對抗資料不平衡。
        """
        v = votes.copy()
        
        # 標註機率丟棄 (Annotation Dropout)：防止模型對單一主導情緒過擬合
        total_votes = int(v.sum())
        n_drop = max(1, int(total_votes * 0.2))
        for _ in range(n_drop):
            drop_pool = [i for i in self.majority_classes if v[i] > 0]
            if not drop_pool:
                break
            drop_idx = np.random.choice(drop_pool)
            v[drop_idx] -= 1.0

        d = v / (v.sum() + 1e-8)
        
        if self.split == "Train":
            d_prime = d * self.w_norm
            d_prime_normalized = d_prime / (d_prime.sum() + 1e-8)
        else:
            d_prime_normalized = d
        
        return torch.tensor(d_prime_normalized, dtype=torch.float32)

    def _get_secondary_distribution(self, sec_votes):
        v = sec_votes.copy()
        d = v / (v.sum() + 1e-8)
        return torch.tensor(d, dtype=torch.float32)

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        w_feat = torch.load(record["feat_path"]).float()  
        label_dist = self._get_target_distribution(record["votes"], self.apply_aug)
        secondary_dist = self._get_secondary_distribution(record["secondary_votes"])
        avd_target = torch.tensor(record["avd"], dtype=torch.float32)
        text = record["text"]
        
        effective_length = w_feat.shape[-1]
        
        # ==========================================
        # 資料增強 (Data Augmentation) - Audio Mixing 
        # 每當遇到多數情緒 (多半是 Neutral 或 Happy) 時，有 50% 機率強行混入一段少數情緒 (Fear, Disgust)
        # ==========================================
        if self.apply_aug and record['consensus_label'] in self.majority_classes and random.random() < 0.5:
            min_record = random.choices(self.minority_records, weights=self.record_weights, k=1)[0]
            min_feat = torch.load(min_record["feat_path"]).float()
            min_dist = self._get_target_distribution(min_record["votes"], self.apply_aug)
            min_sec_dist = self._get_secondary_distribution(min_record["secondary_votes"])
            min_avd = torch.tensor(min_record["avd"], dtype=torch.float32)

            if random.random() < 0.5:
                feat_first, feat_second = w_feat, min_feat
                text_first, text_second = text, min_record["text"]
            else:
                feat_first, feat_second = min_feat, w_feat
                text_first, text_second = min_record["text"], text

            mix_type = random.choice(["silence", "overlap"])
            
            if mix_type == "silence":
                silence_len = random.randint(0, 100)
                mel_bins = feat_first.shape[0] 
                silence = torch.zeros((mel_bins, silence_len), dtype=feat_first.dtype)
                mixed_feat = torch.cat([feat_first, silence, feat_second], dim=-1)
            else:
                overlap_len = random.randint(0, 100)
                if overlap_len > 0 and feat_first.shape[-1] > overlap_len and feat_second.shape[-1] > overlap_len:
                    front = feat_first[:, :-overlap_len]
                    overlap_zone = (feat_first[:, -overlap_len:] + feat_second[:, :overlap_len]) / 2.0
                    back = feat_second[:, overlap_len:]
                    mixed_feat = torch.cat([front, overlap_zone, back], dim=-1)
                else:
                    mixed_feat = torch.cat([feat_first, feat_second], dim=-1)

            w_feat = mixed_feat
            effective_length = w_feat.shape[-1]  

            # 文本、標籤機率均等地按 1:1 比例混合
            text = text_first + " </s> " + text_second

            label_dist = (label_dist + min_dist) / 2.0
            secondary_dist = (secondary_dist + min_sec_dist) / 2.0
            avd_target = (avd_target + min_avd) / 2.0

        target_frames = 1500
        current_frames = w_feat.shape[-1]

        effective_length = min(effective_length, target_frames)
        
        if current_frames > target_frames:
            w_feat = w_feat[:, :target_frames]
        elif current_frames < target_frames:
            pad_len = target_frames - current_frames
            w_feat = F.pad(w_feat, (0, pad_len))

        t_in = self.roberta_tokenizer(
            text, padding='max_length', max_length=128,
            truncation=True, return_tensors="pt"
        )
        
        return (
            w_feat, 
            t_in.input_ids.squeeze(0), 
            t_in.attention_mask.squeeze(0), 
            label_dist,
            secondary_dist,
            avd_target,
            torch.tensor(effective_length, dtype=torch.long)
        )