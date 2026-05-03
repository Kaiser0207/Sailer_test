import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
   "MeanPooling",
    "AttentiveStatisticsPooling",
    "AttentionPoolingBatched",
]


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1) # (batch_size, )
        feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
        return feat_lens
        
    def forward(self, x, mask):
        raise NotImplementedError
    
class MeanPooling(Pooling):
    """
    Mean pooling over the time dimension
    """
    def __init__(self, input_size=None):
        super().__init__()
        self._indim = input_size
    
    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim)
        """
        batch_size, max_len, feat_dim = xs.size()
        
        # Get actual sequence lengths
        feat_lens = self.compute_length_from_mask(mask)
        
        # Create attention mask (1 for valid positions, 0 for padded)
        attention_mask = torch.zeros(batch_size, max_len, device=xs.device)
        for i, length in enumerate(feat_lens):
            attention_mask[i, :length] = 1.0
        
        # Expand mask to match feature dimensions
        attention_mask = attention_mask.unsqueeze(2)  # (batch_size, T, 1)
        
        # Apply mask and compute sum
        masked_xs = xs * attention_mask
        summed = torch.sum(masked_xs, dim=1)  # (batch_size, feat_dim)
        
        # Compute mean by dividing by actual lengths
        lengths = torch.tensor(feat_lens, device=xs.device, dtype=xs.dtype).unsqueeze(1)
        mean_pooled = summed / lengths.clamp(min=1)  # Avoid division by zero
        
        return mean_pooled

class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)



# Alternative implementation with batch processing (more efficient)
class AttentionPoolingBatched(Pooling):
    """
    AttentionPooling with batch processing - more efficient than iterating
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.attention_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim)
        """
        batch_size, max_len, feat_dim = xs.size()
        
        # Convert mask to attention mask (0 for padded, 1 for valid)
        # Assuming mask contains actual lengths, create attention mask
        feat_lens = self.compute_length_from_mask(mask)
        attention_mask = torch.zeros(batch_size, max_len, device=xs.device)
        for i, length in enumerate(feat_lens):
            attention_mask[i, :length] = 1.0
        
        # Compute attention scores
        h = torch.tanh(self.attention_linear(xs))  # (batch_size, T, feat_dim)
        scores = torch.matmul(h, self.attention).squeeze(dim=2)  # (batch_size, T)
        
        # Mask padded positions with -inf before softmax
        scores = scores.masked_fill(attention_mask == 0, -float('inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1).unsqueeze(2)  # (batch_size, T, 1)
        
        # Apply attention weights
        pooled = torch.sum(xs * attn_weights, dim=1)  # (batch_size, feat_dim)
        
        return pooled