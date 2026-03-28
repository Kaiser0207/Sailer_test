import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

class MSP_Podcast_Dataset(Dataset):
    def __init__(self, data_dir, split="Train", roberta_tokenizer=None, apply_aug=True):
        self.data_dir = data_dir
        self.split = split
        self.roberta_tokenizer = roberta_tokenizer
        self.apply_aug = apply_aug and split == "Train"
        
        self.feat_dir = os.path.join(data_dir, "Whisper_Features_30s")
        self.transcripts_dir = os.path.join(data_dir, "Transcripts")
        
        self.consensus_path = os.path.join(data_dir, "Labels", "labels_consensus.csv")
        self.detailed_path = os.path.join(data_dir, "Labels", "labels_detailed.csv")
        
        self.emotion_map_short = {'N': 0, 'A': 1, 'S': 2, 'H': 3, 'F': 4, 'D': 5, 'U': 6, 'C': 7}
        self.emotion_map_long = {'Neutral': 0, 'Angry': 1, 'Sad': 2, 'Happy': 3, 'Fear': 4, 'Disgust': 5, 'Surprise': 6, 'Contempt': 7}
        
        self.majority_classes = [0, 1, 2, 3]
        self.minority_classes = [4, 5, 6, 7]
        
        print("從 labels_detailed.csv 聚合真實得票分佈...")
        self.vote_dict = self._build_vote_dictionary()
        
        self.data_records = self._load_data()
        
        # Distribution Re-weighting
        class_counts = np.zeros(8)
        for r in self.data_records:
            class_counts[r['consensus_label']] += 1
        q = class_counts / class_counts.sum()
        w = 1.0 / (q + 1e-8)
        self.w_norm = w / w.sum()
        
        # Minority 反比權重抽樣（論文要求）
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
        df_detail = pd.read_csv(self.detailed_path)
        df_detail = df_detail[df_detail['EmoClass_Major'].isin(self.emotion_map_long.keys())]
        
        vote_dict = {}
        grouped = df_detail.groupby(['FileName', 'EmoClass_Major']).size().unstack(fill_value=0)
        
        for filename, row in grouped.iterrows():
            votes = np.zeros(8, dtype=np.float32)
            for emo_str, count in row.items():
                if emo_str in self.emotion_map_long:
                    idx = self.emotion_map_long[emo_str]
                    votes[idx] = count
            vote_dict[filename] = votes
        return vote_dict

    def _load_data(self):
        records = []
        df_consensus = pd.read_csv(self.consensus_path)

        if self.split == "Train":
            df_split = df_consensus[df_consensus['Split_Set'] == self.split]
            df_valid = df_split[df_split['EmoClass'].isin(self.emotion_map_short.keys())]
            df_no_agree = df_split[~df_split['EmoClass'].isin(self.emotion_map_short.keys())]
            df_use = pd.concat([df_valid, df_no_agree])
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
            
            votes = self.vote_dict.get(filename, None)
            if votes is None or votes.sum() == 0:
                # No Agreement 或找不到投票：退回 uniform 或單票
                emo_key = row.get('EmoClass', None)
                votes = np.zeros(8, dtype=np.float32)
                if emo_key in self.emotion_map_short:
                    votes[self.emotion_map_short[emo_key]] = 1.0
                else:
                    votes = np.ones(8, dtype=np.float32) / 8.0  

            # No Agreement 樣本 consensus_label 設為最高票的類別
            consensus_label = self.emotion_map_short.get(row.get('EmoClass', None), int(np.argmax(votes)))

            records.append({
                "feat_path": feat_path,
                "text": text,
                "consensus_label": consensus_label,
                "votes": votes
            })
        return records

    def _get_target_distribution(self, votes, is_training):
        v = votes.copy()
        
        # Annotation Dropout 
        if is_training and v.sum() > 1:
            total_votes = int(v.sum())
            n_drop = max(1, int(total_votes * 0.2))
            for _ in range(n_drop):
                drop_pool = [i for i in self.majority_classes if v[i] > 0]
                if not drop_pool:
                    break
                drop_idx = np.random.choice(drop_pool)
                v[drop_idx] -= 1.0

        d = v / (v.sum() + 1e-8)
        
        # Distribution Re-weighting
        if self.split == "Train":
            d_prime = d * self.w_norm
            d_prime_normalized = d_prime / (d_prime.sum() + 1e-8)
        else:
            d_prime_normalized = d
        
        return torch.tensor(d_prime_normalized, dtype=torch.float32)

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        w_feat = torch.load(record["feat_path"]).float()  
        label_dist = self._get_target_distribution(record["votes"], self.apply_aug)
        text = record["text"]
        
        if self.apply_aug and record['consensus_label'] in self.majority_classes and random.random() < 0.5:
            min_record = random.choices(self.minority_records, weights=self.record_weights, k=1)[0]
            min_feat = torch.load(min_record["feat_path"]).float()
            min_dist = self._get_target_distribution(min_record["votes"], self.apply_aug)

            if random.random() < 0.5:
                feat_first, feat_second = w_feat, min_feat
                text_first, text_second = text, min_record["text"]
            else:
                feat_first, feat_second = min_feat, w_feat
                text_first, text_second = min_record["text"], text

            mix_type = random.choice(["silence", "overlap"])
            
            if mix_type == "silence":
                silence_len = random.randint(50, 150)
                mel_bins = feat_first.shape[0] 
                silence = torch.zeros((mel_bins, silence_len), dtype=feat_first.dtype)
                mixed_feat = torch.cat([feat_first, silence, feat_second], dim=-1)
            else:
                overlap_len = random.randint(50, 150)
                front = feat_first[:, :-overlap_len]
                overlap_zone = (feat_first[:, -overlap_len:] + feat_second[:, :overlap_len]) / 2.0
                back = feat_second[:, overlap_len:]
                mixed_feat = torch.cat([front, overlap_zone, back], dim=-1)

            w_feat = mixed_feat

            text = text_first + " </s> " + text_second

            label_dist = (label_dist + min_dist) / 2.0

        target_frames = 3000
        current_frames = w_feat.shape[-1]
        
        if current_frames > target_frames:
            w_feat = w_feat[:, :target_frames]
        elif current_frames < target_frames:
            pad_len = target_frames - current_frames
            w_feat = F.pad(w_feat, (0, pad_len))

        t_in = self.roberta_tokenizer(
            text, padding='max_length', max_length=128,
            truncation=True, return_tensors="pt"
        )
        return w_feat, t_in.input_ids.squeeze(0), t_in.attention_mask.squeeze(0), label_dist