"""
Ir divinkas — SYNAPTIC Lab
SGD System — Estimateur ICA Complexe (Complex Independent Component Analysis)

Implémentation de FastICA pour signaux complexes (circularité non supposée, mais optimisé pour deflation).
Reference: Bingham & Hyvarinen, "A Fast Fixed-Point Algorithm for Independent Component Analysis of Complex Valued Signals"
"""

import numpy as np
import scipy.linalg

def power_normalize(w):
    """Normalise un vecteur complexe."""
    return w / np.linalg.norm(w)

def _complex_whitening(X, n_components=None):
    """
    Blanchiment de données complexes.
    X: (n_features aka n_sensors, n_samples)
    Retourne: X_white, whitening_matrix, dewhitening_matrix
    """
    n_sensors, n_samples = X.shape
    
    # 1. Centrage
    mean = X.mean(axis=1, keepdims=True)
    X_centered = X - mean

    # 2. Covariance
    cov = (X_centered @ X_centered.conj().T) / n_samples
    
    # 3. EVD
    u, s, vh = np.linalg.svd(cov)
    
    if n_components is None:
        n_components = n_sensors

    # 4. Matrice de blanchiment V = D^(-1/2) * U^H
    # Ordre décroissant des valeurs propres par défaut avec svd
    d_inv_sqrt = np.diag(1.0 / np.sqrt(s[:n_components]))
    V = d_inv_sqrt @ u[:, :n_components].conj().T
    
    X_white = V @ X_centered
    
    # Matrice de dé-blanchiment (inverse de V) -> U * D^(1/2)
    V_inv = u[:, :n_components] @ np.diag(np.sqrt(s[:n_components]))
    
    return X_white, V, V_inv, mean

class ComplexFastICA:
    """
    Implémentation légère de FastICA Complexe.
    """
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.W_ = None
        self.V_ = None
        self.mean_ = None
        
    def _g_logcosh(self, z):
        """
        Fonction de contraste G(y) = log(0.1 + y) avec y = |z|^2
        g(z) = derivee complexe de G(|z|^2) par rapport à z*
        
        Pour FastICA complexe généralisé :
        On cherche les extrémas de E[G(|w^H x|^2)].
        Update rule (Fixed Point) pour w:
          w+ = E[ x * (w^H x)^* * g(|w^H x|^2) ] - E[ g(|w^H x|^2) + |w^H x|^2 * g'(|w^H x|^2) ] * w
        
        Mais une version plus simple pour sources circulaires (Bingham algo):
          w+ = E[ x * (w^H x)^* * g(|w^H x|^2) ] - E[ g_prime_part ] * w
          
        Contraste log: G(y) = log(0.1 + y)
        g(y) = 1 / (0.1 + y)   (dérivée par rapport à y)
        g'(y) = -1 / (0.1 + y)^2
        """
        # y = |z|^2
        y = np.abs(z)**2
        a1 = 0.1
        
        # g pour l'update
        g = 1.0 / (a1 + y)
        
        # derivee de g par rapport à y
        dg = -1.0 / ((a1 + y)**2)
        
        return g, dg

    def fit_transform(self, X):
        """
        X: shape (n_sensors, n_samples)
        Returns: S (n_components, n_samples)
        """
        n_sensors, n_samples = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # 1. Blanchiment
        X_white, V, V_inv, mean = _complex_whitening(X, self.n_components)
        self.V_ = V
        self.mean_ = mean
        self.V_inv_ = V_inv
        
        n_comp = X_white.shape[0]
        W = np.zeros((n_comp, n_comp), dtype=complex)
        
        # 2. Deflation (extraction composante par composante)
        for i in range(n_comp):
            # Init vecteur aléatoire
            w = rng.randn(n_comp) + 1j * rng.randn(n_comp)
            w = power_normalize(w)
            
            for it in range(self.max_iter):
                w_old = w
                
                # Projection : z = w^H * X
                # Attention : numpy dot sur complexes conjugue le premier argument si on utilise vdot, mais @ ne conjugue pas.
                # w.conj().T @ X -> (1, N)
                z = w.conj().T @ X_white 
                
                # Non-linéarité
                g, dg = self._g_logcosh(z)
                
                # Fixed Point Update
                # Terme 1: E[ x * (w^H x)^* * g ]
                # z.conj() = (w^H x)^* = w^T x^* (scalaire)
                # On veut mean(X * z_conj * g)
                
                # X_white * (z.conj() * g) -> broadcast
                # shape (n_comp, n_samples)
                term1 = (X_white * (z.conj() * g)).mean(axis=1)
                
                # Terme 2: E[ g + |z|^2 * dg ] * w (approx pour circulaire)
                # Pour le cas général complexe :
                # E[ g + y * dg ]
                y = np.abs(z)**2
                term2_coef = (g + y * dg).mean()
                term2 = term2_coef * w
                
                w = term1 - term2
                
                # Decorrelation (Gram-Schmidt)
                # w = w - sum( (w . w_j^H) w_j )
                if i > 0:
                    # projection sur les w déjà trouvés
                    # W[:i] est (i, n_comp), w est (n_comp,)
                    # scalar_prods = W[:i].conj() @ w
                    # W[:i].T @ scalar_prods
                    for j in range(i):
                        w -= (W[j].conj() @ w) * W[j]
                
                w = power_normalize(w)
                
                # Convergence check (colinearité)
                # |w^H * w_old| -> 1
                cos_sim = np.abs(w.conj() @ w_old)
                if 1 - cos_sim < self.tol:
                    break
            
            W[i, :] = w
            
        self.W_ = W
        
        # Reconstruction des sources
        # S = W * X_white
        S = W.conj() @ X_white
        
        return S

def estimate_ica(
    X: np.ndarray,
    n_components: int = None,
    max_iter: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sépare les sources via Complex FastICA.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Signal reçu (M antennes, N échantillons).
    n_components : int or None
        Nombre de composantes à extraire.
    max_iter : int
        Nombre maximum d'itérations.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    S_estimated : np.ndarray, shape (n_components, N)
        Sources estimées.
    W_unmixing : np.ndarray, shape (n_components, M)
        Matrice de démélange globale (incluant blanchiment).
        S_est = W_unmixing @ (X - mean)
    W_white : np.ndarray
        Matrice de séparation dans l'espace blanchi.
    """
    cica = ComplexFastICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state
    )
    
    S_est = cica.fit_transform(X)
    
    # Matrice globale : W_ica @ V_whiten
    # Attention au sens : S = W_ica.conj() @ (V @ X)
    # Donc W_global = W_ica.conj() @ V
    
    W_white = cica.W_
    W_global = W_white.conj() @ cica.V_
    
    return S_est, W_global, W_white

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    # Adjust sys.path to allow absolute imports from the root if run directly
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from simulator_engine.signal_model import load_config, generate_signal_from_config
    from simulator_engine.channel_model import apply_channel
    import matplotlib.pyplot as plt

    # Test rapide
    cfg = load_config()
    # Force 2 sources pour test
    cfg['signal']['frequencies'] = [1e6, 2e6] 
    cfg['signal']['amplitudes'] = [1.0, 1.0]
    
    t, s_ref, X = generate_signal_from_config(cfg)
    
    # On triche un peu pour créer 2 sources indépendantes pour le test
    # Actuellement generate_signal crée UN signal (une somme de sinusoides)
    # ICA a besoin de sources statistiquement indépendantes.
    # On va créer une 2ème source manuellement
    
    # Source 1 : QPSK
    N = len(t)
    symb1 = (np.random.randint(0, 2, N)*2 - 1) + 1j*(np.random.randint(0, 2, N)*2 - 1)
    # Source 2 : Sinusoide pure
    symb2 = np.exp(1j * 2 * np.pi * 0.05 * np.arange(N))
    
    S_true = np.vstack([symb1, symb2])
    
    # Matrice de mélange arbitraire (complex)
    A = np.array([[1, 0.5j], [0.5, 1]], dtype=complex)
    
    # Mélange
    X_sim = A @ S_true
    
    # Ajout de bruit
    noise = (np.random.randn(*X_sim.shape) + 1j*np.random.randn(*X_sim.shape)) * 0.01
    X_obs = X_sim + noise

    print("Running Complex ICA...")
    S_est, W_glob, _ = estimate_ica(X_obs, n_components=2)
    
    print(f"Original shape: {S_true.shape}")
    print(f"Estimated shape: {S_est.shape}")
    
    # Plot constellations
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].scatter(S_true[0].real, S_true[0].imag, alpha=0.5, s=1)
    axes[0, 0].set_title("Source 1 (True)")
    
    axes[0, 1].scatter(S_true[1].real, S_true[1].imag, alpha=0.5, s=1)
    axes[0, 1].set_title("Source 2 (True)")
    
    axes[1, 0].scatter(S_est[0].real, S_est[0].imag, alpha=0.5, s=1)
    axes[1, 0].set_title("Source 1 (Est)")
    
    axes[1, 1].scatter(S_est[1].real, S_est[1].imag, alpha=0.5, s=1)
    axes[1, 1].set_title("Source 2 (Est)")
    
    plt.tight_layout()
    plt.savefig("ica_test_constellation.png")
    print("✅ Complex ICA testé. Constellations sauvegardées.")
