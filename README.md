# Ad Creative Quality Scorer

Multimodal ad creative quality scoring using **MXNet/GluonCV ResNet-50** for image features, **BiLSTM** over ad copy, learned gating fusion, and **ONNX export** for 40% latency reduction on CPU inference.

---

## Architecture

```
Image features (2048d ResNet-50 pool5)
        │
  [Linear 512 → BN → ReLU → Drop(0.3)]
  [Linear 256 → ReLU]
        │ (256d)
        ├──────────────────────────────┐
                                       │  Learned gating
Ad copy text                           │  gate = σ(W·[img,txt])
        │                              │  fused = [img·g, txt·(1-g)]
  [Embedding(500, 64)]                 │       (384d)
  [BiLSTM(64) × 2 layers]             │
  [Dropout(0.3) → Linear 128]         │
        │ (128d)                       │
        └──────────────────────────────┘
                    │
              [384d fused]
            ┌──────┴──────┐
     [quality head]  [category head]
     Linear 64→1     Linear 64→5
     sigmoid         softmax
     regression      5-class clf
```

## Design choices

| Choice | Rationale |
|--------|-----------|
| ResNet-50 over ViT | Production-proven, GluonCV pretrained weights, lower compute cost at inference |
| BiLSTM over Transformer for text | Ad copy is ≤20 tokens — BiLSTM captures bidirectional context at much lower cost than attention |
| Late fusion with gating | Learns which modality is more informative per creative type (text-heavy → text gate higher; product photo → image gate higher) |
| ONNX export | ~40% CPU latency reduction vs MXNet runtime on identical hardware |
| Multi-task training | Category prediction acts as auxiliary task, prevents quality head overfitting |

---

## Quickstart

### 1. Train on Google Colab (recommended — MXNet GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saitejasrivilli/ad-creative-scorer/blob/main/notebooks/train_colab.ipynb)

Runtime → Change runtime type → T4 GPU → Run all (~10 min)

### 2. Serve locally (after training)

```bash
pip install fastapi uvicorn onnxruntime numpy
python serving/api.py
```

### 3. Score a creative

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "ad_id": 1,
    "ad_copy": "Shop Laptops — Free Shipping Today",
    "bid_cpm": 3.0,
    "predicted_ctr": 0.05
  }'
```

### 4. Build C++ postprocessor

```bash
g++ -O2 -std=c++17 -shared -fPIC \
  -o postprocess/scorer.so \
  postprocess/scorer.cpp
```

---

## Project structure

```
ad-creative-scorer/
├── model/
│   └── multitask.py          # Architecture definition + NumPy reference
├── export/
│   └── onnx_export.py        # ONNX export + p50/p95/p99 benchmark
├── postprocess/
│   ├── scorer.cpp            # C++ batch score aggregation + eCPM integration
│   └── scorer.py             # Python ctypes wrapper
├── serving/
│   └── api.py                # FastAPI scoring endpoint
├── data/
│   └── synthetic_data.py     # Synthetic ad creative dataset generator
├── notebooks/
│   └── train_colab.ipynb     # One-click Colab training
└── requirements.txt
```

---

## Metrics (20K creatives, 20 epochs, T4 GPU)

| Metric | Score |
|--------|-------|
| Quality score correlation (r) | 0.71+ |
| Category accuracy | 0.85+ |
| MXNet p99 latency | ~5ms |
| ONNX p99 latency | ~3ms |
| Latency reduction | ~40% |

---

## References

- [GluonCV: A Deep Learning Toolkit for Computer Vision](https://arxiv.org/abs/1812.01585)
- [MXNet: A Flexible and Efficient ML Library](https://arxiv.org/abs/1512.01274)
- [ONNX: Open Neural Network Exchange](https://onnx.ai/)

---

## License

MIT
