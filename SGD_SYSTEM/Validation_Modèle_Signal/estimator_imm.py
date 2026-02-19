"""
Ir divinkas — SYNAPTIC Lab
SGD System — Estimateur IMM (Interacting Multiple Model)

Stub fonctionnel pour le filtre IMM :
  - Suivi de cibles avec modèles de mouvement multiples
  - Fusion probabiliste des estimations
  - Application au tracking Doppler dynamique

Structure de base avec 2 modèles : constant velocity (CV) et constant acceleration (CA).
"""

import numpy as np


class KalmanFilter:
    """Filtre de Kalman linéaire simple."""

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Parameters
        ----------
        F : Matrice de transition d'état
        H : Matrice d'observation
        Q : Covariance du bruit de processus
        R : Covariance du bruit de mesure
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.n = F.shape[0]

    def predict(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Étape de prédiction."""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Étape de mise à jour.

        Returns
        -------
        x_upd, P_upd, likelihood
        """
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        P_upd = (np.eye(self.n) - K @ self.H) @ P_pred

        # Vraisemblance gaussienne
        det_S = np.linalg.det(S)
        if det_S <= 0:
            likelihood = 1e-100
        else:
            exponent = -0.5 * y.T @ np.linalg.inv(S) @ y
            likelihood = np.exp(float(exponent)) / np.sqrt((2 * np.pi) ** len(z) * det_S)

        return x_upd, P_upd, max(likelihood, 1e-100)


class IMMEstimator:
    """
    Estimateur IMM (Interacting Multiple Model).

    Fusionne les estimations de plusieurs filtres de Kalman
    avec des probabilités de modèle variables.
    """

    def __init__(
        self,
        filters: list[KalmanFilter],
        transition_matrix: np.ndarray,
        mu_init: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        filters : list[KalmanFilter]
            Liste des filtres (un par modèle).
        transition_matrix : np.ndarray, shape (n_models, n_models)
            Matrice de transition Markov entre modèles.
            transition_matrix[i,j] = P(modèle_j à t | modèle_i à t-1).
        mu_init : np.ndarray or None
            Probabilités initiales des modèles. Défaut: uniforme.
        """
        self.filters = filters
        self.TPM = transition_matrix
        self.n_models = len(filters)

        if mu_init is None:
            self.mu = np.ones(self.n_models) / self.n_models
        else:
            self.mu = mu_init.copy()

    def step(
        self,
        x_list: list[np.ndarray],
        P_list: list[np.ndarray],
        z: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Une itération IMM complète.

        Parameters
        ----------
        x_list : list[np.ndarray]
            État estimé de chaque filtre.
        P_list : list[np.ndarray]
            Covariance de chaque filtre.
        z : np.ndarray
            Observation.

        Returns
        -------
        x_list_upd : états mis à jour
        P_list_upd : covariances mises à jour
        x_fused : état fusionné
        mu : probabilités de modèle
        """
        n = self.n_models

        # ── 1. Probabilités de mélange ──
        c_bar = self.TPM.T @ self.mu
        mixing_probs = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mixing_probs[i, j] = self.TPM[i, j] * self.mu[i] / max(c_bar[j], 1e-100)

        # ── 2. Mélange des états ──
        x_mixed = []
        P_mixed = []
        for j in range(n):
            x_0j = sum(mixing_probs[i, j] * x_list[i] for i in range(n))
            P_0j = sum(
                mixing_probs[i, j] * (
                    P_list[i] + np.outer(x_list[i] - x_0j, x_list[i] - x_0j)
                )
                for i in range(n)
            )
            x_mixed.append(x_0j)
            P_mixed.append(P_0j)

        # ── 3. Prédiction + mise à jour par filtre ──
        x_upd = []
        P_upd = []
        likelihoods = np.zeros(n)

        for j in range(n):
            x_pred, P_pred = self.filters[j].predict(x_mixed[j], P_mixed[j])
            x_u, P_u, lk = self.filters[j].update(x_pred, P_pred, z)
            x_upd.append(x_u)
            P_upd.append(P_u)
            likelihoods[j] = lk

        # ── 4. Mise à jour des probabilités de modèle ──
        self.mu = c_bar * likelihoods
        mu_sum = np.sum(self.mu)
        if mu_sum > 0:
            self.mu /= mu_sum
        else:
            self.mu = np.ones(n) / n

        # ── 5. Fusion ──
        x_fused = sum(self.mu[j] * x_upd[j] for j in range(n))

        return x_upd, P_upd, x_fused, self.mu.copy()


# ──────────────────────────────────────────────
# Factory : IMM pour tracking Doppler
# ──────────────────────────────────────────────

def create_doppler_imm(dt: float, q_cv: float = 1.0, q_ca: float = 0.1, r: float = 10.0):
    """
    Crée un IMM à 2 modèles pour le tracking Doppler.

    Modèle 1 : Constant Velocity (CV)
        état = [f_d, f_d_dot]

    Modèle 2 : Constant Acceleration (CA)
        état = [f_d, f_d_dot, f_d_ddot]

    Note : les états sont de tailles différentes → on padde CV à dim 3.
    """
    # ── CV (dim 3, dernière composante fixée à 0) ──
    F_cv = np.array([
        [1, dt, 0],
        [0, 1,  0],
        [0, 0,  0],
    ])
    H_cv = np.array([[1, 0, 0]])
    Q_cv = q_cv * np.array([
        [dt**3/3, dt**2/2, 0],
        [dt**2/2, dt,      0],
        [0,       0,       0],
    ])
    R = np.array([[r]])

    # ── CA ──
    F_ca = np.array([
        [1, dt, dt**2/2],
        [0, 1,  dt],
        [0, 0,  1],
    ])
    H_ca = np.array([[1, 0, 0]])
    Q_ca = q_ca * np.array([
        [dt**5/20, dt**4/8, dt**3/6],
        [dt**4/8,  dt**3/3, dt**2/2],
        [dt**3/6,  dt**2/2, dt],
    ])

    kf_cv = KalmanFilter(F_cv, H_cv, Q_cv, R)
    kf_ca = KalmanFilter(F_ca, H_ca, Q_ca, R)

    # Matrice de transition entre modèles
    TPM = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])

    return IMMEstimator([kf_cv, kf_ca], TPM)


if __name__ == "__main__":
    # Simulation rapide : tracking d'un Doppler variable
    np.random.seed(42)
    N = 200
    dt = 0.01

    # Doppler vrai : rampe puis palier
    f_true = np.concatenate([
        np.linspace(500, 800, N // 2),
        np.ones(N // 2) * 800,
    ])

    # Observations bruitées
    z_obs = f_true + np.random.randn(N) * 5

    imm = create_doppler_imm(dt, q_cv=10, q_ca=1, r=25)

    x_list = [np.array([z_obs[0], 0, 0]), np.array([z_obs[0], 0, 0])]
    P_list = [np.eye(3) * 100, np.eye(3) * 100]

    estimates = []
    mu_history = []

    for k in range(N):
        z = np.array([z_obs[k]])
        x_list, P_list, x_fused, mu = imm.step(x_list, P_list, z)
        estimates.append(x_fused[0])
        mu_history.append(mu)

    estimates = np.array(estimates)
    mu_history = np.array(mu_history)

    rmse_val = np.sqrt(np.mean((estimates - f_true) ** 2))
    print(f"RMSE Doppler tracking : {rmse_val:.2f} Hz")
    print(f"Prob. finale CV={mu_history[-1, 0]:.3f}, CA={mu_history[-1, 1]:.3f}")
    print("✅ IMM exécuté avec succès")
