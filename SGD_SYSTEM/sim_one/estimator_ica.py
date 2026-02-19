"""
Ir divinkas — SYNAPTIC Lab
SGD System — Estimateur ICA (Independent Component Analysis)

Stub fonctionnel utilisant FastICA de scikit-learn pour :
  - Séparation de sources à partir du signal multi-antenne
  - Estimation des composantes indépendantes
"""

import numpy as np


def estimate_ica(
    X: np.ndarray,
    n_components: int = None,
    max_iter: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sépare les sources via FastICA.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Signal reçu (M antennes, N échantillons).
    n_components : int or None
        Nombre de composantes à extraire. None = M.
    max_iter : int
        Nombre maximum d'itérations.
    random_state : int
        Graine aléatoire pour reproductibilité.

    Returns
    -------
    S_estimated : np.ndarray, shape (n_components, N)
        Sources estimées.
    W : np.ndarray, shape (n_components, M)
        Matrice de démélange.
    """
    from sklearn.decomposition import FastICA

    if n_components is None:
        n_components = X.shape[0]

    # FastICA attend (n_samples, n_features) → transposer
    # On travaille sur les parties réelles (FastICA ne supporte pas le complexe nativement)
    X_real = np.vstack([np.real(X), np.imag(X)])

    ica = FastICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        whiten="unit-variance",
    )

    S_est = ica.fit_transform(X_real.T).T  # (n_components, N)
    W = ica.components_

    return S_est, W


if __name__ == "__main__":
    from signal_model import load_config, generate_signal_from_config
    from channel_model import apply_channel

    cfg = load_config()
    t, s, X = generate_signal_from_config(cfg)
    X_noisy, _ = apply_channel(X, cfg)

    S_est, W = estimate_ica(X_noisy, n_components=3)

    print(f"Sources estimées : {S_est.shape}")
    print(f"Matrice W        : {W.shape}")
    print("✅ ICA exécuté avec succès")
