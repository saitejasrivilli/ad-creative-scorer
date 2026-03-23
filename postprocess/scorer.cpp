/*
 * Ad Creative Score Postprocessor
 * Batched score aggregation, percentile normalization, and eCPM
 * integration for the downstream auction ranking pipeline.
 *
 * Compile:
 *   g++ -O2 -std=c++17 -shared -fPIC -o scorer.so scorer.cpp
 *
 * Design:
 *   - Raw quality scores from ONNX are in [0,1] but not calibrated
 *   - Percentile normalization maps scores to [0,1] relative to
 *     a rolling window of recent scores (prevents score drift)
 *   - eCPM integration: final_ecpm = bid_cpm * ctr * quality_score^alpha
 *   - Alpha controls how much creative quality amplifies/dampens eCPM
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <deque>
#include <mutex>
#include <stdexcept>

// ── Score normalization ────────────────────────────────────────────────────

class PercentileNormalizer {
public:
    explicit PercentileNormalizer(size_t window_size = 10000)
        : window_size_(window_size) {}

    // Update rolling window with new scores
    void update(const std::vector<float>& scores) {
        std::lock_guard<std::mutex> lock(mu_);
        for (float s : scores) {
            window_.push_back(s);
            if (window_.size() > window_size_) {
                window_.pop_front();
            }
        }
    }

    // Normalize a batch of scores to [0,1] based on rolling percentiles
    std::vector<float> normalize(const std::vector<float>& scores) {
        std::lock_guard<std::mutex> lock(mu_);
        if (window_.empty()) return scores;

        std::vector<float> sorted_window(window_.begin(), window_.end());
        std::sort(sorted_window.begin(), sorted_window.end());

        std::vector<float> normalized(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            // Find percentile rank of score in rolling window
            auto it = std::lower_bound(sorted_window.begin(), sorted_window.end(), scores[i]);
            size_t rank = std::distance(sorted_window.begin(), it);
            normalized[i] = static_cast<float>(rank) / sorted_window.size();
        }
        return normalized;
    }

    size_t window_size() const { return window_.size(); }

private:
    size_t window_size_;
    std::deque<float> window_;
    std::mutex mu_;
};

// ── eCPM integration ───────────────────────────────────────────────────────

struct AuctionCandidate {
    int    ad_id;
    float  bid_cpm;
    float  predicted_ctr;
    float  quality_score;   // raw [0,1] from ONNX
};

struct AdjustedCandidate {
    int   ad_id;
    float quality_normalized;  // percentile-normalized quality
    float adjusted_ecpm;       // bid * ctr * quality^alpha
    float rank_score;          // final ranking score
};

/*
 * Compute adjusted eCPM incorporating creative quality.
 *
 * Formula: adjusted_ecpm = bid_cpm * predicted_ctr * quality_score^alpha
 *
 * alpha=0 → quality has no effect (pure eCPM)
 * alpha=1 → quality linearly scales eCPM
 * alpha=0.3 → quality has moderate amplification effect
 *
 * Tuned via A/B testing against revenue and user experience metrics.
 */
std::vector<AdjustedCandidate> compute_adjusted_ecpm(
    const std::vector<AuctionCandidate>& candidates,
    const std::vector<float>& normalized_scores,
    float alpha = 0.3f
) {
    if (candidates.size() != normalized_scores.size()) {
        throw std::invalid_argument("candidates and scores must have same size");
    }

    std::vector<AdjustedCandidate> results(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        float q = normalized_scores[i];
        float quality_factor = std::pow(q + 1e-6f, alpha);
        float adj_ecpm = c.bid_cpm * c.predicted_ctr * quality_factor;

        results[i] = {
            c.ad_id,
            q,
            adj_ecpm,
            adj_ecpm,  // rank_score = adjusted_ecpm by default
        };
    }

    return results;
}

// ── Batch processor ────────────────────────────────────────────────────────

class BatchScoreProcessor {
public:
    explicit BatchScoreProcessor(float alpha = 0.3f, size_t window = 10000)
        : alpha_(alpha), normalizer_(window) {}

    /*
     * Process a batch of auction candidates:
     * 1. Update rolling normalizer with raw scores
     * 2. Percentile-normalize scores
     * 3. Compute adjusted eCPM
     * 4. Sort by adjusted eCPM descending
     *
     * Returns candidates sorted by rank_score descending.
     */
    std::vector<AdjustedCandidate> process(
        const std::vector<AuctionCandidate>& candidates
    ) {
        // Extract raw quality scores
        std::vector<float> raw_scores(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            raw_scores[i] = candidates[i].quality_score;
        }

        // Update normalizer
        normalizer_.update(raw_scores);

        // Normalize
        std::vector<float> norm_scores = normalizer_.normalize(raw_scores);

        // Compute adjusted eCPM
        auto results = compute_adjusted_ecpm(candidates, norm_scores, alpha_);

        // Sort by rank_score descending
        std::sort(results.begin(), results.end(),
            [](const AdjustedCandidate& a, const AdjustedCandidate& b) {
                return a.rank_score > b.rank_score;
            });

        return results;
    }

    size_t normalizer_window_size() const {
        return normalizer_.window_size();
    }

private:
    float alpha_;
    PercentileNormalizer normalizer_;
};

// ── C API (for Python ctypes binding) ─────────────────────────────────────

extern "C" {

/*
 * Process a batch of scores.
 * Input arrays are parallel: ad_ids[i], bid_cpms[i], ctrs[i], quality[i]
 * Output written to out_ecpm[i] sorted descending by adjusted eCPM.
 *
 * Returns number of valid outputs written.
 */
int process_batch(
    const int*   ad_ids,
    const float* bid_cpms,
    const float* ctrs,
    const float* quality_scores,
    int          n,
    float        alpha,
    float*       out_ecpm,
    int*         out_ad_ids
) {
    if (n <= 0) return 0;

    static BatchScoreProcessor processor(alpha);

    std::vector<AuctionCandidate> candidates(n);
    for (int i = 0; i < n; ++i) {
        candidates[i] = {ad_ids[i], bid_cpms[i], ctrs[i], quality_scores[i]};
    }

    auto results = processor.process(candidates);

    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        out_ecpm[i]   = results[i].adjusted_ecpm;
        out_ad_ids[i] = results[i].ad_id;
    }

    return static_cast<int>(results.size());
}

} // extern "C"
