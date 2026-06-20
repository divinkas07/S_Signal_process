"""
Independent Component Analysis (ICA) – Blind source separation.
Separates mixed signals received on the antenna array into
independent components, isolating the desired source from interference.

Uses FastICA algorithm.
"""

import numpy as np


class ICAProcessor:
    """
    FastICA implementation for blind source separation.
    Separates N_sources from N_antennas mixed signals.
    """

    def __init__(self, n_components: int = None, max_iter: int = 200, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self._metrics = {}
        self._mixing_matrix = None
        self._unmixing_matrix = None

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: extract signals from array data.
        
        Expects data to contain:
            'array_signal': ndarray of shape (n_antennas, n_samples)
        
        Adds to data:
            'ica_sources': separated source signals
            'ica_mixing':  estimated mixing matrix
        """
        X = data.get("array_signal")
        if X is None:
            self._metrics = {"status": "no_array_signal"}
            return data

        n_components = self.n_components or min(X.shape[0], 4)
        sources, W = self.fastica(X, n_components)

        data["ica_sources"] = sources
        data["ica_mixing"] = W
        self._metrics = {
            "n_sources": n_components,
            "status": "success",
            "source_powers_db": [
                float(10 * np.log10(np.mean(np.abs(s)**2) + 1e-20))
                for s in sources
            ],
        }
        return data

    def fastica(self, X: np.ndarray, n_components: int) -> tuple:
        """
        FastICA algorithm.

        Args:
            X: Mixed signals, shape (n_channels, n_samples)
            n_components: Number of independent components to extract.

        Returns:
            (sources, W): Separated sources and unmixing matrix.
        """
        n_channels, n_samples = X.shape
        n_components = min(n_components, n_channels)

        # 1. Center the data
        X_mean = X.mean(axis=1, keepdims=True)
        X_centered = X - X_mean

        # 2. Whiten the data (PCA + scaling)
        # Use real part for eigendecomposition of complex data
        if np.iscomplexobj(X_centered):
            Cov = (X_centered @ X_centered.conj().T) / n_samples
        else:
            Cov = (X_centered @ X_centered.T) / n_samples

        eigenvalues, eigenvectors = np.linalg.eigh(Cov)

        # Select top components (descending order)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        D = np.diag(1.0 / np.sqrt(eigenvalues[idx] + 1e-10))
        E = eigenvectors[:, idx]
        K = D @ E.conj().T  # Whitening matrix

        X_white = K @ X_centered  # (n_components, n_samples)

        # 3. FastICA iteration
        W = np.random.randn(n_components, n_components)
        if np.iscomplexobj(X_white):
            W = W.astype(complex)

        # QR orthogonalization
        W, _ = np.linalg.qr(W)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            for p in range(n_components):
                w = W[p, :]

                # Non-linearity: g(u) = tanh(u) for real, |u|^2 * u for complex
                if np.iscomplexobj(X_white):
                    wx = w @ X_white
                    g = np.abs(wx)**2 * wx
                    gp = 3 * np.abs(wx)**2
                else:
                    wx = w @ X_white
                    g = np.tanh(wx)
                    gp = 1 - np.tanh(wx)**2

                w_new = (X_white @ g.conj()) / n_samples - np.mean(gp) * w

                # Deflation: orthogonalize against previous components
                for j in range(p):
                    w_new -= (w_new @ W[j].conj()) * W[j]

                # Normalize
                norm = np.linalg.norm(w_new)
                if norm > 0:
                    W[p, :] = w_new / norm
                else:
                    W[p, :] = np.random.randn(n_components)
                    W[p, :] /= np.linalg.norm(W[p, :])

            # Check convergence
            diff = np.max(np.abs(np.abs(np.diag(W @ W_old.conj().T)) - 1))
            if diff < self.tol:
                break

        # 4. Recover sources
        sources = W @ X_white  # (n_components, n_samples)
        self._unmixing_matrix = W @ K
        self._mixing_matrix = np.linalg.pinv(self._unmixing_matrix)

        return sources, self._unmixing_matrix

    def get_metrics(self) -> dict:
        return self._metrics
