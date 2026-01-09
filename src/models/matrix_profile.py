import numpy as np

class MatrixProfile:
    def __init__(self, window = 100):
        self.window = window
        self.model_name = 'MatrixProfile'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        try:
            import stumpy
        except ImportError as exc:
            raise ImportError(
                "stumpy is required for MatrixProfile; install with `theseus[matrix-profile]`."
            ) from exc

        X = np.asarray(X, dtype=float)
        window = int(self.window)
        if window <= 0:
            raise ValueError("window must be a positive integer")

        # `stumpy.stump` returns an array whose first column is the matrix profile.
        self.profile_ = stumpy.stump(X, m=window)
        self.decision_scores_ = self.profile_[:, 0]
        return self
    
    def top_k_discords(self, k=5):
        scores = np.asarray(getattr(self, "decision_scores_", []), dtype=float)
        if scores.size == 0:
            return []
        k = max(0, min(int(k), scores.size))
        return np.argsort(scores)[-k:][::-1].tolist()
