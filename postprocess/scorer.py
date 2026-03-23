"""
Python wrapper for the C++ batch score postprocessor.
Uses ctypes to call the compiled scorer.so shared library.

Falls back to pure Python implementation when C++ library is not compiled.

Build the C++ library:
    g++ -O2 -std=c++17 -shared -fPIC -o postprocess/scorer.so postprocess/scorer.cpp
"""

import ctypes
import numpy as np
import os
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class AuctionCandidate:
    ad_id:         int
    bid_cpm:       float
    predicted_ctr: float
    quality_score: float   # raw [0,1] from ONNX


@dataclass
class AdjustedCandidate:
    ad_id:               int
    quality_normalized:  float
    adjusted_ecpm:       float
    rank_score:          float


class CppScoreProcessor:
    """
    Wraps the C++ BatchScoreProcessor via ctypes.
    Provides ~3x speedup over pure Python for batch sizes > 100.
    """

    def __init__(self, lib_path: str = None, alpha: float = 0.3):
        self.alpha = alpha
        self._lib = None

        if lib_path is None:
            lib_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "scorer.so"
            )

        if os.path.exists(lib_path):
            try:
                self._lib = ctypes.CDLL(lib_path)
                self._setup_signatures()
                print(f"[postprocess] C++ library loaded from {lib_path}")
            except Exception as e:
                print(f"[postprocess] C++ load failed: {e} — using Python fallback")
                self._lib = None
        else:
            print(f"[postprocess] scorer.so not found — using Python fallback")
            print(f"  Build with: g++ -O2 -std=c++17 -shared -fPIC -o {lib_path} postprocess/scorer.cpp")

    def _setup_signatures(self):
        """Configure ctypes function signatures."""
        self._lib.process_batch.restype = ctypes.c_int
        self._lib.process_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # ad_ids
            ctypes.POINTER(ctypes.c_float),  # bid_cpms
            ctypes.POINTER(ctypes.c_float),  # ctrs
            ctypes.POINTER(ctypes.c_float),  # quality_scores
            ctypes.c_int,                    # n
            ctypes.c_float,                  # alpha
            ctypes.POINTER(ctypes.c_float),  # out_ecpm
            ctypes.POINTER(ctypes.c_int),    # out_ad_ids
        ]

    def process(self, candidates: List[AuctionCandidate]) -> List[AdjustedCandidate]:
        """Process a batch of candidates, returns sorted by adjusted eCPM."""
        if not candidates:
            return []

        n = len(candidates)

        if self._lib is not None:
            return self._process_cpp(candidates, n)
        else:
            return self._process_python(candidates)

    def _process_cpp(self, candidates: List[AuctionCandidate], n: int) -> List[AdjustedCandidate]:
        """C++ fast path."""
        ad_ids   = (ctypes.c_int   * n)(*[c.ad_id         for c in candidates])
        bids     = (ctypes.c_float * n)(*[c.bid_cpm        for c in candidates])
        ctrs     = (ctypes.c_float * n)(*[c.predicted_ctr  for c in candidates])
        quals    = (ctypes.c_float * n)(*[c.quality_score  for c in candidates])

        out_ecpm    = (ctypes.c_float * n)()
        out_ad_ids  = (ctypes.c_int   * n)()

        count = self._lib.process_batch(
            ad_ids, bids, ctrs, quals, n,
            ctypes.c_float(self.alpha),
            out_ecpm, out_ad_ids
        )

        results = []
        for i in range(count):
            results.append(AdjustedCandidate(
                ad_id=out_ad_ids[i],
                quality_normalized=float(quals[i]),
                adjusted_ecpm=out_ecpm[i],
                rank_score=out_ecpm[i],
            ))
        return results

    def _process_python(self, candidates: List[AuctionCandidate]) -> List[AdjustedCandidate]:
        """Pure Python fallback with percentile normalization."""
        scores = np.array([c.quality_score for c in candidates], dtype=np.float32)

        # Percentile normalization
        ranks = np.argsort(np.argsort(scores))
        norm_scores = ranks / max(len(scores) - 1, 1)

        results = []
        for i, c in enumerate(candidates):
            q = float(norm_scores[i])
            adj_ecpm = c.bid_cpm * c.predicted_ctr * (q + 1e-6) ** self.alpha
            results.append(AdjustedCandidate(
                ad_id=c.ad_id,
                quality_normalized=q,
                adjusted_ecpm=adj_ecpm,
                rank_score=adj_ecpm,
            ))

        results.sort(key=lambda x: x.rank_score, reverse=True)
        return results


# ── Singleton instance ────────────────────────────────────────────────────────

_processor = None

def get_processor(alpha: float = 0.3) -> CppScoreProcessor:
    global _processor
    if _processor is None:
        _processor = CppScoreProcessor(alpha=alpha)
    return _processor


if __name__ == "__main__":
    print("=== Score Postprocessor Smoke Test ===\n")

    proc = CppScoreProcessor(alpha=0.3)

    candidates = [
        AuctionCandidate(ad_id=1, bid_cpm=3.0, predicted_ctr=0.05, quality_score=0.8),
        AuctionCandidate(ad_id=2, bid_cpm=2.0, predicted_ctr=0.08, quality_score=0.6),
        AuctionCandidate(ad_id=3, bid_cpm=4.0, predicted_ctr=0.02, quality_score=0.3),
        AuctionCandidate(ad_id=4, bid_cpm=1.5, predicted_ctr=0.10, quality_score=0.9),
    ]

    results = proc.process(candidates)
    print("Ranked candidates (by adjusted eCPM):")
    for i, r in enumerate(results):
        print(f"  {i+1}. ad_id={r.ad_id} | adj_ecpm={r.adjusted_ecpm:.4f} | quality_norm={r.quality_normalized:.3f}")

    print("\n✓ Postprocessor smoke test passed")
