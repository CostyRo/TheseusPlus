import numpy as np


class ARIMACPD:
    def __init__(self, order=(1, 1, 1), trend="n", score="abs_resid"):
        self.order = tuple(order)
        self.trend = trend
        self.score = score
        self.model_name = "ARIMA"

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float).reshape(-1)
        n = int(X.size)
        if n == 0:
            self.decision_scores_ = np.array([], dtype=float)
            return self

        x_min = float(np.min(X))
        x_max = float(np.max(X))
        if not (np.isfinite(x_min) and np.isfinite(x_max)):
            raise ValueError("X contains non-finite values")
        if x_min == x_max:
            self.decision_scores_ = np.zeros(n, dtype=float)
            return self

        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError(
                "ARIMACPD requires statsmodels (install with `uv sync --extra feature-extraction`)."
            ) from exc

        try:
            model = ARIMA(
                X,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit()
            resid = np.asarray(res.resid, dtype=float).reshape(-1)
        except Exception:
            self.decision_scores_ = np.zeros(n, dtype=float)
            return self

        resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
        if resid.size == 0:
            self.decision_scores_ = np.zeros(n, dtype=float)
            return self

        if resid.size < n:
            resid = np.concatenate([np.full(n - resid.size, resid[0], dtype=float), resid])
        elif resid.size > n:
            resid = resid[-n:]

        if self.score in (None, "abs", "abs_resid"):
            scores = np.abs(resid)
        elif self.score in ("sq", "sq_resid"):
            scores = resid**2
        else:
            raise ValueError("score must be one of: 'abs_resid', 'sq_resid'")

        self.decision_scores_ = np.asarray(scores, dtype=float)
        return self

    def top_k_change_points(self, k=5):
        scores = np.asarray(getattr(self, "decision_scores_", []), dtype=float)
        if scores.size == 0:
            return []
        k = max(0, min(int(k), scores.size))
        return np.argsort(scores)[-k:][::-1].tolist()

