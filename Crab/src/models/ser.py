import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalEmotionClassifierDeep(nn.Module):
    def __init__(
        self,
        features1_dim=1024,  
        features2_dim=768, 
        fusion_hidden_dim=512,
        num_emotions=8,
        dropout=0.5,
        contrastive_dim=512  # Dimension for contrastive embeddings
    ):
        super().__init__()
        
        # Separate modality processing
        self.speech_projection = nn.Linear(features1_dim, fusion_hidden_dim)
        self.text_projection = nn.Linear(features2_dim, fusion_hidden_dim)
        
        self.speech_norm = nn.LayerNorm(fusion_hidden_dim)
        self.text_norm = nn.LayerNorm(fusion_hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

        # GRU layers
        self.speech_gru = nn.GRU(
            fusion_hidden_dim,
            fusion_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.text_gru = nn.GRU(
            fusion_hidden_dim,
            fusion_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Cross-modal attention
        self.speech_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 1, dropout=dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 1, dropout=dropout, batch_first=True)
        
        # Simple attention pooling
        self.speech_attn = nn.Linear(fusion_hidden_dim * 2, 1)
        self.text_attn = nn.Linear(fusion_hidden_dim * 2, 1)
        
        # Contrastive embedding MLPs for frame-level features
        self.speech_contrastive_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, contrastive_dim)
        )
        
        self.text_contrastive_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, contrastive_dim)
        )
        
        # Contrastive embedding MLPs for pooled features
        self.speech_pooled_contrastive_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, contrastive_dim)
        )
        
        self.text_pooled_contrastive_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, contrastive_dim)
        )
        
        # Fusion contrastive embedding MLP
        self.fusion_contrastive_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, contrastive_dim)
        )
        
        self.classifier_fc1 = nn.Linear(fusion_hidden_dim * 4, fusion_hidden_dim)
        self.classifier_relu = nn.ReLU()
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier_fc2 = nn.Linear(fusion_hidden_dim, num_emotions)
        self.layer_norm = nn.LayerNorm(fusion_hidden_dim * 4)
    
    def attention_pool(self, features, attention_layer, mask=None):
        # features: [batch, seq_len, hidden]
        
        # Calculate attention scores
        attn_weights = attention_layer(features)  # [batch, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention
        weighted_features = features * attn_weights
        pooled = weighted_features.sum(dim=1)  # [batch, hidden]
        
        return pooled
    
    def forward(self, features1, features2, return_embeddings=False):
        # Project both modalities to same dimension
        speech_proj = self.speech_projection(features1)  # [batch, seq_len, fusion_hidden_dim]
        text_proj = self.text_projection(features2)    # [batch, seq_len, fusion_hidden_dim]
        
        speech_proj = self.speech_norm(speech_proj)
        text_proj = self.text_norm(text_proj)
        
        # Pass through GRUs
        speech_hidden, _ = self.speech_gru(speech_proj)  # [batch, seq_len, fusion_hidden_dim*2]
        text_hidden, _ = self.text_gru(text_proj)        # [batch, seq_len, fusion_hidden_dim*2]

        speech_attended, _ = self.speech_attention(
            speech_hidden, text_hidden, text_hidden
        )
        text_attended, _ = self.text_attention(
            text_hidden, speech_hidden, speech_hidden
        )
        
        # Combine attended features
        speech_final = speech_hidden + speech_attended
        text_final = text_hidden + text_attended
        
        # Simple attention pooling with masks
        speech_pooled = self.attention_pool(speech_final, self.speech_attn)
        text_pooled = self.attention_pool(text_final, self.text_attn)
        
        # Concatenate pooled representations
        concatenated = torch.cat([speech_pooled, text_pooled], dim=-1)  # [batch, fusion_hidden_dim*4]
        
        # Layer norm and classify
        normalized = self.layer_norm(concatenated)

        # Pass through classifier layers
        hidden = self.classifier_fc1(normalized)
        hidden_activated = self.classifier_relu(hidden)
        hidden_dropped = self.classifier_dropout(hidden_activated)
        logits = self.classifier_fc2(hidden_dropped)
        
        if return_embeddings:
            # Get contrastive embeddings
            
            # Frame-level embeddings (mean pooled)
            speech_frame_mean = speech_proj.mean(dim=1)  # [batch, fusion_hidden_dim]
            speech_frame_emb = self.speech_contrastive_mlp(speech_frame_mean)
            
            text_frame_mean = text_proj.mean(dim=1)  # [batch, fusion_hidden_dim]
            text_frame_emb = self.text_contrastive_mlp(text_frame_mean)
            
            # Pooled embeddings
            speech_pooled_emb = self.speech_pooled_contrastive_mlp(speech_pooled)
            
            text_pooled_emb = self.text_pooled_contrastive_mlp(text_pooled)
            
            # Fusion embedding
            fusion_emb = self.fusion_contrastive_mlp(hidden)
            
            return logits, {
                'speech_frame_emb': speech_frame_emb,
                'text_frame_emb': text_frame_emb,
                'speech_pooled_emb': speech_pooled_emb,
                'text_pooled_emb': text_pooled_emb,
                'fusion_emb': fusion_emb,
                'normalized': normalized
            }
        else:
            return logits

