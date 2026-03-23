"""
Multimodal Ad Creative Quality Scorer — PyTorch implementation.

Architecture:
  Image branch:  ResNet-50 pool5 features (2048d) -> projection 512->256
  Text branch:   Embedding(500,64) -> BiLSTM(64) -> 128d
  Fusion:        Learned gating weights -> 384d
  Output heads:
    quality_score  regression  [0,1]  weight=1.0
    category       5-class clf        weight=0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

IMAGE_FEATURE_DIM = 2048
TEXT_VOCAB_SIZE   = 500
TEXT_EMBED_DIM    = 64
MAX_TEXT_LEN      = 20
LSTM_HIDDEN       = 64
NUM_CATEGORIES    = 5
FUSION_DIM        = 384


class ImageBranch(nn.Module):
    def __init__(self, input_dim=IMAGE_FEATURE_DIM, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.proj(x)


class TextBranch(nn.Module):
    def __init__(self, vocab_size=TEXT_VOCAB_SIZE, embed_dim=TEXT_EMBED_DIM,
                 hidden=LSTM_HIDDEN, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, 128)
    def forward(self, tokens):
        emb = self.embed(tokens)
        out, _ = self.bilstm(emb)
        pooled = out.mean(dim=1)
        return self.proj(self.drop(pooled))


class GatingFusion(nn.Module):
    def __init__(self, img_dim=256, txt_dim=128):
        super().__init__()
        self.gate_layer = nn.Linear(img_dim + txt_dim, 1)
    def forward(self, img, txt):
        gate = torch.sigmoid(self.gate_layer(torch.cat([img, txt], dim=-1)))
        return torch.cat([img * gate, txt * (1 - gate)], dim=-1)


class CreativeScorer(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, dropout=0.3):
        super().__init__()
        self.image_branch = ImageBranch(dropout=dropout)
        self.text_branch  = TextBranch(dropout=dropout)
        self.fusion       = GatingFusion()
        self.quality_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
        self.category_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_categories))

    def forward(self, image_features, text_tokens):
        img = self.image_branch(image_features)
        txt = self.text_branch(text_tokens)
        fused = self.fusion(img, txt)
        return self.quality_head(fused), self.category_head(fused)

    def compute_loss(self, q_pred, c_pred, q_true, c_true, qw=1.0, cw=0.5):
        q_loss = F.mse_loss(q_pred.squeeze(), q_true)
        c_loss = F.cross_entropy(c_pred, c_true)
        total  = qw * q_loss + cw * c_loss
        return total, {"loss": total.item(), "q_loss": q_loss.item(), "c_loss": c_loss.item()}


if __name__ == "__main__":
    model = CreativeScorer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    img = torch.randn(8, IMAGE_FEATURE_DIM)
    txt = torch.randint(0, TEXT_VOCAB_SIZE, (8, MAX_TEXT_LEN))
    q, c = model(img, txt)
    print(f"Quality: {q.shape}, Category: {c.shape}")
    print("✓ Smoke test passed")
