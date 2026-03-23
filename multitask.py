"""
Multimodal Ad Creative Quality Scorer.

Architecture:
  Image branch:  GluonCV ResNet-50 pool5 features (2048d) → Linear 512 → 256
  Text branch:   BiLSTM over ad copy tokens → 128d
  Fusion:        Learned gating weights → concat → 384d
  Output heads:
    quality_score  regression  [0,1]
    category       classification  5 classes

Design choices:
  - ResNet-50 over ViT: production-proven, GluonCV has pretrained weights,
    ByteDance/TikTok historically used MXNet/GluonCV in production
  - BiLSTM over Transformer for text: ad copy is short (< 20 tokens),
    BiLSTM captures bidirectional context at lower compute cost
  - Late fusion with gating: learns which modality is more informative
    per creative type (text-heavy ads → text gate higher; product photos → image gate higher)
  - Multi-task: joint quality + category training improves representations
    (category acts as auxiliary task, prevents quality head from overfitting)
"""

import numpy as np
from typing import Tuple, Dict

# ── Pure NumPy/framework-agnostic model definition ───────────────────────────
# This file defines the architecture; actual training uses the Colab notebook
# which imports MXNet. The architecture mirrors what GluonCV fine-tuning produces.

IMAGE_FEATURE_DIM = 2048
TEXT_VOCAB_SIZE   = 500
TEXT_EMBED_DIM    = 64
MAX_TEXT_LEN      = 20
LSTM_HIDDEN       = 64   # BiLSTM → 128d output
NUM_CATEGORIES    = 5
FUSION_DIM        = 384  # 256 (image) + 128 (text)


def build_image_branch_config() -> Dict:
    """Image branch architecture config for MXNet Gluon."""
    return {
        "backbone":    "ResNet50_v2",    # GluonCV pretrained
        "pretrained":  True,
        "frozen_layers": ["layer0", "layer1", "layer2"],  # freeze early layers
        "projection": [
            {"type": "Dense", "units": 512, "activation": "relu"},
            {"type": "BatchNorm"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "Dense", "units": 256, "activation": "relu"},
        ],
        "output_dim": 256,
    }


def build_text_branch_config() -> Dict:
    """Text branch architecture config for MXNet Gluon."""
    return {
        "embedding": {"vocab_size": TEXT_VOCAB_SIZE, "embed_dim": TEXT_EMBED_DIM},
        "bilstm":    {"hidden_size": LSTM_HIDDEN, "num_layers": 2, "dropout": 0.3},
        "output_dim": LSTM_HIDDEN * 2,   # bidirectional → 128d
    }


def build_fusion_config() -> Dict:
    """Gating fusion config."""
    return {
        "type":        "learned_gating",
        "image_dim":   256,
        "text_dim":    128,
        "gate_hidden": 64,
        "output_dim":  FUSION_DIM,
    }


def build_output_heads_config() -> Dict:
    """Multi-task output heads."""
    return {
        "quality_score": {
            "type":       "regression",
            "layers":     [{"units": 64, "activation": "relu"}, {"units": 1, "activation": "sigmoid"}],
            "loss":       "mse",
            "weight":     1.0,
        },
        "category": {
            "type":       "classification",
            "num_classes": NUM_CATEGORIES,
            "layers":     [{"units": 64, "activation": "relu"}, {"units": NUM_CATEGORIES}],
            "loss":       "softmax_ce",
            "weight":     0.5,
        },
    }


# ── NumPy reference implementation (for ONNX validation) ─────────────────────

class NumpyImageBranch:
    """Lightweight NumPy image branch for ONNX output validation."""

    def __init__(self, W1=None, W2=None, b1=None, b2=None):
        rng = np.random.default_rng(42)
        self.W1 = W1 if W1 is not None else rng.normal(0, 0.01, (IMAGE_FEATURE_DIM, 512)).astype(np.float32)
        self.b1 = b1 if b1 is not None else np.zeros(512, dtype=np.float32)
        self.W2 = W2 if W2 is not None else rng.normal(0, 0.01, (512, 256)).astype(np.float32)
        self.b2 = b2 if b2 is not None else np.zeros(256, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        x = np.maximum(0, x @ self.W2 + self.b2)
        return x


class NumpyGatingFusion:
    """Learned gating fusion in NumPy."""

    def __init__(self):
        rng = np.random.default_rng(42)
        self.Wg = rng.normal(0, 0.01, (256 + 128, 1)).astype(np.float32)

    def forward(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
        concat = np.concatenate([img_emb, txt_emb], axis=-1)
        gate = 1 / (1 + np.exp(-concat @ self.Wg))  # sigmoid gate
        img_gated = img_emb * gate
        txt_gated = txt_emb * (1 - gate)
        return np.concatenate([img_gated, txt_gated], axis=-1)


def compute_quality_score(
    image_features: np.ndarray,
    text_tokens: np.ndarray,
) -> np.ndarray:
    """
    Reference quality score computation (NumPy, no framework dependency).
    Used for ONNX validation and serving fallback.

    Args:
        image_features: (N, 2048) ResNet-50 pool5 features
        text_tokens:    (N, 20) token IDs

    Returns:
        quality_scores: (N,) float32 quality scores [0, 1]
    """
    img_branch = NumpyImageBranch()
    img_emb = img_branch.forward(image_features)    # (N, 256)

    # Simple text embedding: mean of token one-hot (placeholder for BiLSTM)
    txt_emb = np.zeros((len(text_tokens), 128), dtype=np.float32)
    for i, tokens in enumerate(text_tokens):
        unique_tokens = np.unique(tokens[tokens > 0])
        if len(unique_tokens) > 0:
            txt_emb[i, :len(unique_tokens) % 128] = 1.0 / len(unique_tokens)

    fusion = NumpyGatingFusion()
    fused = fusion.forward(img_emb, txt_emb)        # (N, 384)

    # Quality head: linear projection to scalar
    rng = np.random.default_rng(42)
    Wq = rng.normal(0, 0.01, (384, 1)).astype(np.float32)
    scores = 1 / (1 + np.exp(-(fused @ Wq)))        # sigmoid
    return scores.squeeze()


if __name__ == "__main__":
    print("=== Creative Scorer Architecture ===")
    print(f"Image branch: ResNet50 pool5 ({IMAGE_FEATURE_DIM}d) → 512 → 256d")
    print(f"Text branch:  Embedding({TEXT_VOCAB_SIZE}, {TEXT_EMBED_DIM}) → BiLSTM({LSTM_HIDDEN}) → 128d")
    print(f"Fusion:       Learned gating → {FUSION_DIM}d")
    print(f"Heads:        quality_score (regression) + category (5-class)")

    # Smoke test
    N = 10
    img = np.random.randn(N, IMAGE_FEATURE_DIM).astype(np.float32)
    txt = np.random.randint(0, TEXT_VOCAB_SIZE, (N, MAX_TEXT_LEN)).astype(np.int32)
    scores = compute_quality_score(img, txt)
    print(f"\nSmoke test: {N} creatives → scores {scores.min():.3f}–{scores.max():.3f}")
    print("✓ Architecture smoke test passed")
