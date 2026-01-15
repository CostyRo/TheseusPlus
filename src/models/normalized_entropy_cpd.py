import math

import numpy as np


class NormalizedEntropyCPD:
    def __init__(self, window=100, bins="ln", score="delta"):
        self.window = window
        self.bins = bins
        self.score = score
        self.model_name = "NECPD"

    def _resolve_bins(self, window):
        bins = self.bins
        if isinstance(bins, (int, np.integer)):
            return int(bins)
        if bins in (None, "ln", "log"):
            return int(round(math.log(max(2, int(window)))))
        if bins == "sqrt":
            return int(round(math.sqrt(max(2, int(window)))))
        raise ValueError("bins must be an int or one of: 'ln', 'sqrt'")

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float).reshape(-1)
        n = int(X.size)
        if n == 0:
            self.entropy_ = np.array([], dtype=float)
            self.decision_scores_ = np.array([], dtype=float)
            return self

        window = int(self.window)
        if window < 2:
            raise ValueError("window must be >= 2")
        if window > n:
            window = n

        k = max(2, self._resolve_bins(window))

        x_min = float(np.min(X))
        x_max = float(np.max(X))
        if not (np.isfinite(x_min) and np.isfinite(x_max)):
            raise ValueError("X contains non-finite values")

        if x_min == x_max:
            self.bin_edges_ = np.array([x_min, x_max], dtype=float)
            self.bins_ = 1
            self.window_ = window
            self.entropy_ = np.zeros(n, dtype=float)
            self.decision_scores_ = np.zeros(n, dtype=float)
            return self

        bin_edges = np.linspace(x_min, x_max, num=k + 1, dtype=float)
        bin_idx = np.searchsorted(bin_edges, X, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, k - 1)

        counts = np.bincount(bin_idx[:window], minlength=k).astype(float)
        ent_seq = np.empty(n - window + 1, dtype=float)
        log_k = math.log(k)

        for t in range(window - 1, n):
            if t != window - 1:
                out_bin = int(bin_idx[t - window])
                in_bin = int(bin_idx[t])
                counts[out_bin] -= 1.0
                counts[in_bin] += 1.0

            p = counts / float(window)
            nz = p > 0
            H = -float(np.sum(p[nz] * np.log(p[nz])))
            ent_seq[t - window + 1] = H / log_k

        entropy = np.empty(n, dtype=float)
        entropy[: window - 1] = ent_seq[0]
        entropy[window - 1 :] = ent_seq

        score_mode = self.score
        if score_mode in (None, "delta"):
            scores = np.zeros(n, dtype=float)
            scores[1:] = np.abs(np.diff(entropy))
        elif score_mode == "entropy":
            scores = entropy.copy()
        else:
            raise ValueError("score must be one of: 'delta', 'entropy'")

        self.bin_edges_ = bin_edges
        self.bins_ = k
        self.window_ = window
        self.entropy_ = entropy
        self.decision_scores_ = scores
        return self

    def top_k_change_points(self, k=5):
        scores = np.asarray(getattr(self, "decision_scores_", []), dtype=float)
        if scores.size == 0:
            return []
        k = max(0, min(int(k), scores.size))
        return np.argsort(scores)[-k:][::-1].tolist()
