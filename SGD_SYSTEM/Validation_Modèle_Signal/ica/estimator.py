"""
Ir divinkas — SYNAPTIC Lab
SGD System — FastICA Complexe (Complex Independent Component Analysis)

Ce module implémente l'algorithme FastICA pour les signaux complexes, 
basé sur le travail de Bingham & Hyvarinen.
"""

import numpy as np
import scipy.linalg

def power_normalize(w):
    """Normalise un vecteur complexe pour qu'il ait une puissance unitaire."""
    return w / np.linalg.norm(w)

def complex_whitening(X, n_components=None):
    """
    Blanchiment des données complexes (Whitening).
    Assure que les composantes ont une variance unitaire et sont décorrélées.
    
    X: (n_sensors, n_samples)
    Returns: X_white, whitening_matrix, dewhitening_matrix, mean
    """
    n_sensors, n_samples = X.shape
    
    # 1. Centrage
    mean = X.mean(axis=1, keepdims=True)
    X_centered = X - mean

    # 2. Matrice de Covariance
    cov = (X_centered @ X_centered.conj().T) / n_samples
    
    # 3. Décomposition en valeurs singulières (SVD)
    u, s, vh = np.linalg.svd(cov)
    
    if n_components is None:
        n_components = n_sensors

    # 4. Matrice de blanchiment V = D^(-1/2) * U^H
    d_inv_sqrt = np.diag(1.0 / np.sqrt(s[:n_components] + 1e-12))
    V = d_inv_sqrt @ u[:, :n_components].conj().T
    
    X_white = V @ X_centered
    
    # Matrice de dé-blanchiment (inverse de V)
    V_inv = u[:, :n_components] @ np.diag(np.sqrt(s[:n_components] + 1e-12))
    
    return X_white, V, V_inv, mean

class ComplexFastICA:
    """
    Implémentation de l'algorithme FastICA Complexe.
    Optimisé pour l'extraction séquentielle (deflation).
    """
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.W_ = None  # Matrice de séparation (espace blanchi)
        self.V_ = None  # Matrice de blanchiment
        self.mean_ = None
        
    def _g_logcosh(self, z):
        """
        Fonction de contraste pour la non-linéarité.
        Utilise G(y) = log(0.1 + y) où y = |z|^2.
        """
        y = np.abs(z)**2
        a1 = 0.1
        g = 1.0 / (a1 + y)
        dg = -1.0 / ((a1 + y)**2)
        return g, dg

    def fit_transform(self, X):
        """
        Exécute la séparation de sources.
        X: (n_sensors, n_samples)
        """
        n_sensors, n_samples = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # 1. Blanchiment
        X_white, V, V_inv, mean = complex_whitening(X, self.n_components)
        self.V_ = V
        self.mean_ = mean
        
        n_comp = X_white.shape[0]
        W = np.zeros((n_comp, n_comp), dtype=complex)
        
        # 2. Deflation (extraction composante par composante)
        for i in range(n_comp):
            # Vecteur de poids initial aléatoire
            w = rng.randn(n_comp) + 1j * rng.randn(n_comp)
            w = power_normalize(w)
            
            for it in range(self.max_iter):
                w_old = w
                
                # Projection : z = w^H * X
                z = w.conj().T @ X_white 
                
                # Non-linéarité
                g, dg = self._g_logcosh(z)
                
                # Mise à jour (Fixed Point Update)
                term1 = (X_white * (z.conj() * g)).mean(axis=1)
                y = np.abs(z)**2
                term2_coef = (g + y * dg).mean()
                term2 = term2_coef * w
                
                w = term1 - term2
                
                # Décorrélation (Gram-Schmidt) pour rester orthogonal aux précédents
                if i > 0:
                    for j in range(i):
                        w -= (W[j].conj() @ w) * W[j]
                
                w = power_normalize(w)
                
                # Vérification de convergence
                if 1 - np.abs(w.conj() @ w_old) < self.tol:
                    break
            
            W[i, :] = w
            
        self.W_ = W
        
        # Reconstruction des sources estimées
        S = W.conj() @ X_white
        return S

def estimate_ica(X, n_components=None, max_iter=500, random_state=42):
    """
    Fonction utilitaire pour lancer ICA rapidement.
    Returns: S_est, W_global
    """
    ica = ComplexFastICA(n_components=n_components, max_iter=max_iter, random_state=random_state)
    S_est = ica.fit_transform(X)
    W_global = ica.W_.conj() @ ica.V_
    return S_est, W_global
