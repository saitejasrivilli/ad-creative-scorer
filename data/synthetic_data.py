"""
Synthetic ad creative dataset generator.
Produces image feature vectors + ad copy text + engagement labels
for training the multimodal creative quality scorer.

In production: real ad creatives from the ad serving platform.
Labels: downstream CTR, view-through rate, skip rate.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Ad creative categories
CATEGORIES = ["product", "lifestyle", "text_heavy", "video_thumbnail", "brand"]

# Ad copy templates per category
AD_COPY = {
    "product":   ["Shop {item} — Free Shipping", "Buy {item} Today", "{item} — Best Price"],
    "lifestyle": ["Live Better with {item}", "Discover {item}", "Your {item} Journey Starts Here"],
    "text_heavy":["50% Off {item} — Limited Time", "{item}: Read Reviews | Compare Prices | Buy Now", "Top 10 {item} Deals"],
    "brand":     ["{brand} — Premium {item}", "{brand}: Quality {item} Since 1990", "Trusted {brand}"],
    "video_thumbnail": ["Watch: {item} Review", "{item} — See It In Action", "Video: Best {item} 2024"],
}

ITEMS  = ["laptop", "phone", "headphones", "camera", "watch", "sneakers", "bag", "jacket"]
BRANDS = ["TechPro", "StyleCo", "GadgetHub", "PremiumBrand", "ValueShop"]

# Image feature dimensions (simulates GluonCV ResNet-50 pool5 features)
IMAGE_FEATURE_DIM = 2048
TEXT_VOCAB_SIZE   = 500
MAX_TEXT_LEN      = 20


def _simulate_image_features(category: str, quality: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate ResNet-50 pool5 features for an ad creative.
    High-quality creatives have more structured feature distributions.
    """
    base = rng.normal(0, 1, IMAGE_FEATURE_DIM).astype(np.float32)

    # Category-specific signal in first 64 dims
    cat_idx = CATEGORIES.index(category)
    base[:64] += cat_idx * 0.3

    # Quality signal in dims 64-128
    base[64:128] += quality * 2.0

    # Normalize
    norm = np.linalg.norm(base)
    return base / (norm + 1e-8)


def _tokenize(text: str, vocab_size: int = TEXT_VOCAB_SIZE, max_len: int = MAX_TEXT_LEN) -> np.ndarray:
    """Simple character-hash tokenizer."""
    tokens = text.lower().split()
    ids = [hash(t) % vocab_size for t in tokens[:max_len]]
    # Pad to max_len
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


def _quality_label(
    category: str,
    text_len: int,
    has_cta: bool,
    rng: np.random.Generator,
    noise: float = 0.15,
) -> Tuple[float, float, float]:
    """
    Generate engagement labels (CTR, VTR, skip_rate) correlated with quality signals.
    Returns (quality_score, ctr, skip_rate).
    """
    # Base quality: shorter text + CTA → higher quality
    base = 0.5
    if has_cta:    base += 0.15
    if text_len < 6: base += 0.1
    if category == "product":   base += 0.05
    if category == "text_heavy": base -= 0.1

    quality = float(np.clip(base + rng.normal(0, noise), 0.1, 0.95))
    ctr      = float(np.clip(quality * 0.12 + rng.normal(0, 0.02), 0.001, 0.3))
    skip_rate = float(np.clip(1 - quality + rng.normal(0, 0.1), 0.05, 0.95))

    return quality, ctr, skip_rate


def generate_dataset(
    n_samples: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic ad creative dataset.

    Columns:
        ad_id           int
        category        str   one of CATEGORIES
        ad_copy         str   ad headline text
        quality_score   float [0,1] overall creative quality
        ctr             float click-through rate
        skip_rate       float skip/scroll-past rate
        image_features  list  2048-dim ResNet-50 pool5 (stored as comma-sep string)
        text_tokens     list  MAX_TEXT_LEN token IDs
        has_cta         bool  contains call-to-action phrase
    """
    rng = np.random.default_rng(seed)
    rows = []
    image_feats = []
    text_tok    = []

    for i in range(n_samples):
        category = rng.choice(CATEGORIES)
        item     = rng.choice(ITEMS)
        brand    = rng.choice(BRANDS)
        template = rng.choice(AD_COPY[category])
        ad_copy  = template.format(item=item, brand=brand)
        has_cta  = any(w in ad_copy.lower() for w in ["buy", "shop", "get", "discover", "watch"])
        text_len = len(ad_copy.split())

        quality, ctr, skip_rate = _quality_label(category, text_len, has_cta, rng)
        img_feat = _simulate_image_features(category, quality, rng)
        tokens   = _tokenize(ad_copy)

        rows.append({
            "ad_id":         i,
            "category":      category,
            "ad_copy":       ad_copy,
            "quality_score": round(quality, 4),
            "ctr":           round(ctr, 4),
            "skip_rate":     round(skip_rate, 4),
            "has_cta":       int(has_cta),
            "text_len":      text_len,
        })
        image_feats.append(img_feat)
        text_tok.append(tokens)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} ad creatives")
    print(f"Category distribution:\n{df['category'].value_counts().to_string()}")
    print(f"Mean quality score: {df['quality_score'].mean():.3f}")
    print(f"Mean CTR: {df['ctr'].mean():.4f}")

    return df, np.stack(image_feats), np.stack(text_tok)


def train_val_test_split(
    df: pd.DataFrame,
    image_feats: np.ndarray,
    text_tok: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    n = len(df)
    idx = np.random.default_rng(seed).permutation(n)

    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def split(arr):
        return arr[train_idx], arr[val_idx], arr[test_idx]

    train_df  = df.iloc[train_idx].reset_index(drop=True)
    val_df    = df.iloc[val_idx].reset_index(drop=True)
    test_df   = df.iloc[test_idx].reset_index(drop=True)

    tr_img, va_img, te_img = split(image_feats)
    tr_txt, va_txt, te_txt = split(text_tok)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return (train_df, tr_img, tr_txt), (val_df, va_img, va_txt), (test_df, te_img, te_txt)


if __name__ == "__main__":
    df, img, txt = generate_dataset(n_samples=10000)
    (tr, tr_i, tr_t), (va, va_i, va_t), (te, te_i, te_t) = train_val_test_split(df, img, txt)
    print("\n✓ Dataset generation complete")
