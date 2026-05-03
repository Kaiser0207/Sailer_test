import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss for single GPU/CPU training
    Based on: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embs, labels):
        """
        Args:
            embs: Embeddings tensor of shape [B, D]
            labels: Labels tensor of shape [B]
        
        Returns:
            loss: Scalar loss value
        """
        device = embs.device
        batch_size = embs.size(0)
        
        # Normalize embeddings
        feats = F.normalize(embs, dim=-1, p=2)
        
        # Compute similarity matrix
        logits = torch.matmul(feats, feats.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Apply mask to logits (set self-similarity to very negative value)
        logits = logits - (1 - logits_mask) * 1e9
        
        # Stabilize logits
        logits = stablize_logits(logits)
        
        # Compute ground-truth distribution
        # Each positive pair gets equal probability
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        
        # Compute loss
        loss = compute_cross_entropy(p, logits)
        
        return loss


# Example usage for SER
if __name__ == "__main__":
    # Example: batch of 8 samples with 256-dim embeddings
    batch_size = 8
    embedding_dim = 256
    num_emotions = 4
    
    # Random embeddings and emotion labels
    embeddings = torch.randn(batch_size, embedding_dim)
    emotion_labels = torch.randint(0, num_emotions, (batch_size,))
    
    # Create loss function
    loss_fn = MultiPosConLoss(temperature=0.1)
    
    # Compute loss
    loss = loss_fn(embeddings, emotion_labels)
    print(f"Loss: {loss.item():.4f}")
    
    # For SER, you might want to adjust temperature based on your feature scale
    # Typical ranges: 0.05 - 0.5

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CKALoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def centering(self, K):
        """
        Centers the kernel matrix K using the centering matrix H = I - (1/n)11^T
        Args:
            K: Kernel matrix of shape [n x n]
        Returns:
            Centered kernel matrix
        """
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - (1.0/n) * torch.ones((n, n), device=K.device)
        return H @ K @ H
    
    def forward(self, wav_features, rob_features):
        """
        Compute CKA loss between WavLM and RoBERTa features
        Args:
            wav_features: WavLM features after transformer [batch_size x hidden_dim]
            rob_features: RoBERTa features after transformer [batch_size x hidden_dim]
        Returns:
            CKA loss 
        """
        # Compute Gram matrices
        K = wav_features @ wav_features.T  # [batch_size x batch_size]
        L = rob_features @ rob_features.T  # [batch_size x batch_size]
        
        # Center Gram matrices
        K_centered = self.centering(K)
        L_centered = self.centering(L)
        
        # Compute HSIC
        HSIC_KL = torch.trace(K_centered @ L_centered)
        HSIC_KK = torch.trace(K_centered @ K_centered)
        HSIC_LL = torch.trace(L_centered @ L_centered)
        
        # Compute CKA
        epsilon = 1e-8  # Small constant for numerical stability
        CKA = HSIC_KL / (torch.sqrt(HSIC_KK * HSIC_LL) + epsilon)
        
        # Return loss
        return CKA