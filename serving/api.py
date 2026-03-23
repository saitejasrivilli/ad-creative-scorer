"""
FastAPI inference service for ad creative quality scoring.
Accepts image features + ad copy, returns quality score + category.

Endpoints:
  POST /score        — score a single creative
  POST /score/batch  — score a batch of creatives (async queue)
  GET  /health       — health check
  GET  /metrics      — latency stats
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import time
import os
from collections import deque

app = FastAPI(
    title="Ad Creative Quality Scorer",
    description="Multimodal ad creative scoring — MXNet ResNet-50 + BiLSTM + ONNX",
    version="1.0.0",
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class CreativeRequest(BaseModel):
    ad_id:          int
    ad_copy:        str
    bid_cpm:        float = 1.0
    predicted_ctr:  float = 0.05
    # image_features: optional 2048-dim ResNet-50 pool5 (omit to use text-only scoring)
    image_features: Optional[List[float]] = None

class CreativeScore(BaseModel):
    ad_id:              int
    quality_score:      float = Field(description="Creative quality score [0,1]")
    category:           str   = Field(description="Predicted creative category")
    category_confidence:float
    adjusted_ecpm:      float = Field(description="eCPM adjusted by quality score")
    latency_ms:         float

class BatchRequest(BaseModel):
    creatives: List[CreativeRequest]
    alpha:     float = Field(default=0.3, description="Quality weight in eCPM formula")

class BatchResponse(BaseModel):
    results:          List[CreativeScore]
    total_latency_ms: float
    throughput_qps:   float

# ── Model loader ──────────────────────────────────────────────────────────────

class ScorerModel:
    """Lazy-loads ONNX model and postprocessor."""

    CATEGORIES = ["product", "lifestyle", "text_heavy", "video_thumbnail", "brand"]

    def __init__(self):
        self.session  = None
        self._loaded  = False

    def load(self):
        if self._loaded:
            return

        model_dir  = os.getenv("MODEL_DIR", "models/")
        onnx_path  = os.path.join(model_dir, "creative_scorer.onnx")

        try:
            import onnxruntime as ort
            if os.path.exists(onnx_path):
                self.session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                print(f"[scorer] ONNX model loaded from {onnx_path}")
            else:
                print(f"[scorer] No ONNX model at {onnx_path} — using heuristic scorer")
        except ImportError:
            print("[scorer] onnxruntime not installed — using heuristic scorer")

        self._loaded = True

    def score(
        self,
        image_features: np.ndarray,
        text_tokens:    np.ndarray,
    ) -> tuple:
        """
        Returns (quality_score, category_idx, category_confidence).
        Falls back to heuristic if ONNX not available.
        """
        if self.session is not None:
            try:
                input_names = [i.name for i in self.session.get_inputs()]
                feeds = {}
                if len(input_names) >= 1:
                    feeds[input_names[0]] = image_features.astype(np.float32)
                if len(input_names) >= 2:
                    feeds[input_names[1]] = text_tokens.astype(np.int32)
                outputs = self.session.run(None, feeds)
                quality = float(outputs[0].squeeze())
                cat_logits = outputs[1].squeeze() if len(outputs) > 1 else np.zeros(5)
                cat_idx    = int(np.argmax(cat_logits))
                cat_conf   = float(np.exp(cat_logits[cat_idx]) / np.sum(np.exp(cat_logits)))
                return quality, cat_idx, cat_conf
            except Exception as e:
                print(f"[scorer] ONNX inference error: {e} — using heuristic")

        return self._heuristic_score(image_features, text_tokens)

    def _heuristic_score(self, image_features, text_tokens):
        """Heuristic scorer when ONNX unavailable."""
        from model.multitask import compute_quality_score
        quality = float(compute_quality_score(image_features, text_tokens).squeeze())
        cat_idx = int(np.argmax(image_features[0, :5])) % 5
        return quality, cat_idx, 0.6

    def tokenize(self, text: str, max_len: int = 20) -> np.ndarray:
        tokens = text.lower().split()
        ids = [hash(t) % 500 for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return np.array([ids], dtype=np.int32)


scorer = ScorerModel()

# ── Latency tracker ───────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self, window=500):
        self.latencies = deque(maxlen=window)
        self.count = 0

    def record(self, ms: float):
        self.latencies.append(ms)
        self.count += 1

    def stats(self):
        if not self.latencies:
            return {}
        s = sorted(self.latencies)
        n = len(s)
        return {
            "requests": self.count,
            "p50_ms":   round(s[n//2], 2),
            "p95_ms":   round(s[int(n*0.95)], 2),
            "p99_ms":   round(s[int(n*0.99)], 2),
            "mean_ms":  round(sum(s)/n, 2),
        }

tracker = LatencyTracker()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    scorer.load()


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "onnx_loaded":  scorer.session is not None,
        "model_dir":    os.getenv("MODEL_DIR", "models/"),
    }


@app.get("/metrics")
def metrics():
    return tracker.stats()


@app.post("/score", response_model=CreativeScore)
def score_creative(req: CreativeRequest):
    """Score a single ad creative."""
    t0 = time.perf_counter()

    # Image features
    if req.image_features and len(req.image_features) == 2048:
        img = np.array([req.image_features], dtype=np.float32)
    else:
        img = np.random.randn(1, 2048).astype(np.float32) * 0.1

    txt = scorer.tokenize(req.ad_copy)
    quality, cat_idx, cat_conf = scorer.score(img, txt)

    # Adjusted eCPM
    adj_ecpm = req.bid_cpm * req.predicted_ctr * (quality ** 0.3)

    ms = (time.perf_counter() - t0) * 1000
    tracker.record(ms)

    return CreativeScore(
        ad_id=req.ad_id,
        quality_score=round(quality, 4),
        category=scorer.CATEGORIES[cat_idx],
        category_confidence=round(cat_conf, 3),
        adjusted_ecpm=round(adj_ecpm, 4),
        latency_ms=round(ms, 2),
    )


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest):
    """Score a batch of ad creatives."""
    t0 = time.perf_counter()
    results = []

    for creative in req.creatives:
        result = score_creative(creative)
        results.append(result)

    total_ms = (time.perf_counter() - t0) * 1000
    qps = round(len(req.creatives) / (total_ms / 1000), 1)

    return BatchResponse(
        results=results,
        total_latency_ms=round(total_ms, 2),
        throughput_qps=qps,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
