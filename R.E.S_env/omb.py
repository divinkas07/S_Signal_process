"""
SynapticLEO v2 — Pipeline LEO satellite, formulation mathématique rigoureuse
=============================================================================

Améliorations mathématiques par bloc :

  Bloc 1  SpikeEncoder    — LIF exact (τ_m dV/dt = V_rest − V + R·I)
                             + homéostasie de taux de tir
  Bloc 2  SynapticICA     — blanchiment SVD robuste + négentropie G(y)=log cosh
  Bloc 3  SynapticMUSIC   — covariance FB-averaged (R_fb) + noyau DoG latéral
  Bloc 4  SynapticIMM     — IMM bayésien complet (vraisemblance Gaussienne exacte)
                             + transition Markov + STDP sur poids de mélange
  Bloc 5  SynapticPINN    — intégration RK4 + perturbation J2 + drag atmosphérique
                             + loss physique ||f(x)−ẋ||²
  Bloc 6  MeshRouter      — consensus Gossip pondéré synaptique
                             x_i(t+1) = x_i + ε Σ_j w_ij(x_j − x_i)
  Bloc 7  SynapticAnomaly — Mahalanobis régularisé + divergence KL sur ISI

  STDP (global) : Δw = A± · exp(−|Δt| / τ±) · trace_pré · trace_post

Bande : Ku (12–18 GHz)  |  Cible : LEO ~550 km
"""

from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres bio-physiques (calibrés sur neurone pyramidal cortical)
# ─────────────────────────────────────────────────────────────────────────────

class BioParams:
    # ── LIF membrane ──────────────────────────────────────────────────────────
    TAU_M_MS      = 1.0      # Constante de temps membranaire — rapide pour RF (ms)
    V_REST        = -70.0    # Potentiel de repos (mV)
    V_THRESH      = -60.0    # Seuil de spike — abaissé pour signaux RF (mV)
    V_RESET       = -75.0    # Potentiel de reset post-spike (mV)
    R_M           = 100.0    # Résistance membranaire élevée → fort couplage courant (MΩ)
    T_REFRAC_MS   = 0.5      # Période réfractaire courte (ms)

    # ── STDP paire (Bi & Poo 1998) ────────────────────────────────────────────
    A_PLUS        = 0.010    # Amplitude LTP
    A_MINUS       = 0.012    # Amplitude LTD  (légèrement > A+ pour stabilité)
    TAU_PLUS_MS   = 20.0     # Fenêtre temporelle LTP (ms)
    TAU_MINUS_MS  = 20.0     # Fenêtre temporelle LTD (ms)
    TAU_X_MS      = 20.0     # Décroissance trace pré-synaptique
    TAU_Y_MS      = 20.0     # Décroissance trace post-synaptique
    W_MIN         = 0.0      # Poids minimal
    W_MAX         = 1.0      # Poids maximal

    # ── Homéostasie ───────────────────────────────────────────────────────────
    R_TARGET_HZ   = 5.0      # Taux de tir cible (Hz)
    ETA_HOMEO     = 0.002    # Taux d'adaptation homéostatique du seuil

    # ── LTP trace ─────────────────────────────────────────────────────────────
    TAU_LTP_MS    = 50.0     # Constante de temps trace LTP longue durée

    # ── Synapse conductance ────────────────────────────────────────────────────
    TAU_SYN_MS    = 5.0      # Décroissance conductance synaptique (ms)
    G_MAX         = 0.5      # Conductance maximale normalisée

    # ── Physique RF ────────────────────────────────────────────────────────────
    C_KM_PER_MS   = 299.792  # Vitesse lumière (km/ms)
    KU_FREQ_GHZ   = 14.5     # Fréquence centre Ku band
    LEO_ALT_KM    = 550.0    # Altitude LEO
    EARTH_R_KM    = 6371.0
    GM            = 398600.4418  # Constante gravitationnelle (km³/s²)
    J2            = 1.08263e-3   # Coefficient de perturbation J2
    CD            = 2.2          # Coefficient de traînée atmosphérique
    RHO_500KM     = 1e-13        # Densité atmosphérique à 500 km (kg/m³)
    A_OVER_M      = 0.01         # Rapport surface/masse satellite (m²/kg)


BP = BioParams()


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RFSignal:
    iq_samples:     np.ndarray
    sample_rate_hz: float
    center_freq_hz: float
    snr_db:         float = 0.0
    timestamp_ms:   float = 0.0


@dataclass
class SpikePacket:
    origin_id:    str
    timestamp_ms: float
    spike_times:  List[float]    # ms depuis start
    intensities:  List[float]    # amplitude normalisée [0, 1]
    payload_bits: int = 0
    snr_db:       float = 0.0

    @property
    def mean_intensity(self) -> float:
        return float(np.mean(self.intensities)) if self.intensities else 0.0

    @property
    def isi(self) -> np.ndarray:
        """Inter-Spike Intervals (ms)."""
        if len(self.spike_times) < 2:
            return np.array([])
        t = np.array(self.spike_times)
        return np.diff(t)

    @property
    def firing_rate_hz(self) -> float:
        if len(self.spike_times) < 2:
            return 0.0
        duration_s = (self.spike_times[-1] - self.spike_times[0]) * 1e-3
        return (len(self.spike_times) - 1) / max(duration_s, 1e-6)


@dataclass
class AOAEstimate:
    azimuth_deg:   float
    elevation_deg: float
    confidence:    float
    music_peak:    float
    timestamp_ms:  float = 0.0


@dataclass
class TrajectoryState:
    position_km:  np.ndarray   # [x, y, z] ECI km
    velocity_kms: np.ndarray   # [vx, vy, vz] km/s
    cov:          np.ndarray   # matrice covariance 6×6
    model_probs:  Dict[str, float]
    timestamp_ms: float = 0.0
    predicted:    bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires STDP — formulation paire exponentielle (Bi & Poo 1998)
# ─────────────────────────────────────────────────────────────────────────────

class STDPState:
    """
    Trace pré (x) et post (y) pour le calcul STDP paire.

    Mise à jour continue :
        dx/dt = -x / τ_x        dy/dt = -y / τ_y
    Lors d'un spike pré  : x ← x + 1  →  Δw = +A+ · x · y  (si LTP)
    Lors d'un spike post : y ← y + 1  →  Δw = -A- · x · y  (si LTD)
    """

    def __init__(self, tau_x_ms: float = BP.TAU_X_MS, tau_y_ms: float = BP.TAU_Y_MS):
        self.tau_x = tau_x_ms
        self.tau_y = tau_y_ms
        self.x: float = 0.0   # trace pré
        self.y: float = 0.0   # trace post
        self.last_t: float = 0.0

    def decay(self, t_ms: float) -> None:
        """Décroissance exponentielle des traces depuis last_t jusqu'à t_ms."""
        dt = max(0.0, t_ms - self.last_t)
        self.x *= math.exp(-dt / self.tau_x)
        self.y *= math.exp(-dt / self.tau_y)
        self.last_t = t_ms

    def pre_spike(self, t_ms: float, w: float) -> float:
        """Événement pré-synaptique → LTP si la trace post y > 0."""
        self.decay(t_ms)
        self.x += 1.0
        dw = BP.A_PLUS * self.y   # renforcement proportionnel à l'activité post récente
        return max(BP.W_MIN, min(BP.W_MAX, w + dw))

    def post_spike(self, t_ms: float, w: float) -> float:
        """Événement post-synaptique → LTD si la trace pré x > 0."""
        self.decay(t_ms)
        self.y += 1.0
        dw = -BP.A_MINUS * self.x  # dépression proportionnel à l'activité pré récente
        return max(BP.W_MIN, min(BP.W_MAX, w + dw))


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 1 — SpikeEncoder v2 : LIF exact + homéostasie de taux
# ─────────────────────────────────────────────────────────────────────────────

class SpikeEncoder:
    """
    Neurone Leaky Integrate-and-Fire (LIF) pour encoder le signal RF.

    Équation de membrane (solution exacte entre deux spikes) :
        V(t) = V_inf + (V_0 − V_inf) · exp(−(t − t_0) / τ_m)
        V_inf = V_rest + R_m · I

    Homéostasie du seuil (Triesch 2005) :
        θ(t+1) = θ(t) + η · (r(t) − r_target)
    r(t) = taux de tir courant estimé sur fenêtre glissante.

    Conductance synaptique sortante :
        g(t) = Σ_k g_max · exp(−(t − t_k) / τ_syn)
    → modélise la décroissance du spike sur le canal inter-station.
    """

    def __init__(self, station_id: str):
        self.station_id = station_id

        # État membrane LIF
        self.V_m: float = BP.V_REST                 # Potentiel membranaire (mV)
        self.V_thresh: float = BP.V_THRESH           # Seuil adaptatif (mV)
        self.refractory_until_ms: float = 0.0

        # Traces STDP du codeur
        self.stdp = STDPState()
        self.w_self: float = 0.5                     # Poids autosynaptique initial

        # Homéostasie
        self._spike_times_window: List[float] = []
        self._window_ms: float = 200.0               # Fenêtre d'estimation du taux

        # Conductance synaptique (pour l'émission)
        self._recent_spikes_ms: List[float] = []

        # Métriques
        self.total_energy: float = 0.0
        self.n_spikes: int = 0
        self.n_samples: int = 0

    # ── Résolution LIF exacte ─────────────────────────────────────────────────
    def _lif_update(self, V: float, I_input: float, dt_ms: float) -> float:
        """
        Solution analytique exacte de τ_m dV/dt = V_rest − V + R_m · I.
        V(t+dt) = V_inf + (V − V_inf) · exp(−dt/τ_m)
        V_inf    = V_rest + R_m · I
        """
        V_inf = BP.V_REST + BP.R_M * I_input
        decay = math.exp(-dt_ms / BP.TAU_M_MS)
        return V_inf + (V - V_inf) * decay

    # ── Conductance synaptique sortante ───────────────────────────────────────
    def _synaptic_conductance(self, t_ms: float) -> float:
        """
        g(t) = g_max · w · Σ_k exp(−(t − t_k) / τ_syn)
        Somme sur les spikes récents (nettoyage automatique).
        """
        # Nettoyer les vieux spikes (> 5 τ_syn)
        cutoff = t_ms - 5.0 * BP.TAU_SYN_MS
        self._recent_spikes_ms = [s for s in self._recent_spikes_ms if s > cutoff]
        g = sum(
            BP.G_MAX * self.w_self * math.exp(-(t_ms - s) / BP.TAU_SYN_MS)
            for s in self._recent_spikes_ms
        )
        return min(1.0, g)

    # ── Estimation taux de tir ────────────────────────────────────────────────
    def _firing_rate_hz(self, t_ms: float) -> float:
        cutoff = t_ms - self._window_ms
        self._spike_times_window = [s for s in self._spike_times_window if s > cutoff]
        n = len(self._spike_times_window)
        return n / (self._window_ms * 1e-3)   # Hz

    # ── Homéostasie du seuil ──────────────────────────────────────────────────
    def _homeostatic_update(self, t_ms: float) -> None:
        """
        θ(t+1) = θ(t) + η · (r(t) − r_target)
        Si r(t) > r_target : seuil monte (neurone moins excitable)
        Si r(t) < r_target : seuil descend (neurone plus excitable)
        """
        r = self._firing_rate_hz(t_ms)
        self.V_thresh += BP.ETA_HOMEO * (r - BP.R_TARGET_HZ)
        self.V_thresh = max(BP.V_REST + 5.0, min(BP.V_REST + 30.0, self.V_thresh))

    # ── Encodage principal ────────────────────────────────────────────────────
    def encode(self, signal: RFSignal) -> SpikePacket:
        """
        Encode le signal IQ en SpikePacket via le modèle LIF.
        Le courant d'entrée I(t) est l'enveloppe normalisée du signal IQ.
        """
        envelope = np.abs(signal.iq_samples)
        e_max = np.max(envelope)
        if e_max < 1e-12:
            e_max = 1.0
        envelope_norm = envelope / e_max

        # SNR → échelle du courant : I ∝ SNR_linear^0.5 (bruit en racine)
        snr_lin = 10.0 ** (signal.snr_db / 10.0)
        I_scale = min(10.0, math.sqrt(snr_lin))  # courant en nA équivalent

        # Traitement par blocs de 128 échantillons → ~4 kHz effectif
        # Évite que dt_ms << tau_m (sinon le neurone ne charge jamais)
        BLOCK = 128
        block_dt_ms = 1000.0 * BLOCK / signal.sample_rate_hz
        n_blocks = max(1, len(envelope_norm) // BLOCK)

        spike_times: List[float] = []
        intensities: List[float] = []
        t_ms = signal.timestamp_ms

        for b in range(n_blocks):
            block = envelope_norm[b*BLOCK : (b+1)*BLOCK]
            amp   = float(np.mean(block))   # Énergie moyenne du bloc
            t_ms += block_dt_ms
            self.n_samples += BLOCK

            if t_ms < self.refractory_until_ms:
                self.V_m = BP.V_RESET
                continue

            # Courant d'entrée
            I_in = amp * I_scale

            # Intégration LIF exacte sur block_dt_ms
            self.V_m = self._lif_update(self.V_m, I_in, block_dt_ms)

            # Décision spike
            if self.V_m >= self.V_thresh:
                # ─── Spike émis ──────────────────────────────────────────────
                intensity = (self.V_m - self.V_thresh) / (BP.V_RESET - BP.V_THRESH + 1e-6)
                intensity = max(0.0, min(1.0, abs(intensity)))

                spike_times.append(t_ms)
                intensities.append(intensity)

                # Énergie consommée proportionnelle à (V_thresh - V_rest)
                dE = 0.5 * BP.R_M * ((self.V_thresh - BP.V_REST) ** 2) * 1e-9
                self.total_energy += dE
                self.n_spikes += 1

                # STDP : mise à jour poids autosynaptique
                self.w_self = self.stdp.post_spike(t_ms, self.w_self)

                # Enregistrement pour conductance et homéostasie
                self._recent_spikes_ms.append(t_ms)
                self._spike_times_window.append(t_ms)

                # Reset LIF
                self.V_m = BP.V_RESET
                self.refractory_until_ms = t_ms + BP.T_REFRAC_MS

                # Homéostasie (toutes les 10 ms environ)
                if self.n_spikes % 5 == 0:
                    self._homeostatic_update(t_ms)

        return SpikePacket(
            origin_id=self.station_id,
            timestamp_ms=signal.timestamp_ms,
            spike_times=spike_times,
            intensities=intensities,
            payload_bits=len(spike_times) * 32,
            snr_db=signal.snr_db,
        )

    @property
    def compression_ratio(self) -> float:
        """Ratio d'échantillons non-spike (économie d'énergie de transmission)."""
        return (self.n_samples - self.n_spikes) / max(1, self.n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 2 — SynapticICA v2 : blanchiment SVD + négentropie + inhibition LTP
# ─────────────────────────────────────────────────────────────────────────────

class SynapticICA:
    """
    FastICA avec blanchiment SVD robuste et mesure de négentropie.

    Blanchiment SVD (au lieu d'eigh instable) :
        X = U S Vᵀ  (SVD)
        X_white = U_k · X      (k = n_components premières colonnes)

    Fonction de contraste G(y) = log cosh(y) / a₁  (Hyvärinen 1999)
        g(y)  = tanh(a₁ · y)
        g'(y) = a₁ · (1 − tanh²(a₁ · y))

    Règle de mise à jour :
        w ← E{y · g(y)} − E{g'(y)} · w
        w ← w / ‖w‖

    Inhibition synaptique :
        score_i = var(sᵢ)^0.5 · corr_doppler(sᵢ) · (1 + LTP_i − inhib_i)
    """

    A1 = 1.0    # Paramètre de la fonction G(y) = log cosh(a1 y) / a1

    def __init__(self, n_components: int = 4, max_iter: int = 300, tol: float = 1e-7):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        # Poids synaptiques par source (STDP)
        self.stdp_states = [STDPState() for _ in range(n_components)]
        self.source_weights = np.ones(n_components) / n_components
        self.ltp_traces = np.zeros(n_components)
        self.inhibition = np.zeros(n_components)
        self.excitation = np.zeros(n_components)

        # Matrice de séparation apprise (mise à jour incrémentale)
        self.W: Optional[np.ndarray] = None
        self.W_white: Optional[np.ndarray] = None

    def _svd_whitening(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blanchiment via SVD tronquée, plus stable numériquement que eigh.
        X : (p, n)  →  X_white : (k, n)
        W_white : (k, p)  tel que X_white = W_white · X

        Propriété : Cov(X_white) = I_k
        """
        p, n = X.shape
        k = min(self.n_components, p)

        # Centrage
        X_c = X - X.mean(axis=1, keepdims=True)

        # SVD économique
        U, s, _ = np.linalg.svd(X_c, full_matrices=False)
        U_k = U[:, :k]          # (p, k)
        s_k = s[:k]             # (k,)

        # Matrice de blanchiment : W_white = diag(1/σ) Uᵀ
        s_k_safe = np.maximum(s_k, 1e-10)
        W_white = (U_k / s_k_safe).T   # (k, p)
        X_white = W_white @ X_c        # (k, n)

        return X_white, W_white

    def _fastICA_deflation(self, X_white: np.ndarray) -> np.ndarray:
        """
        FastICA par dégonflage (une composante à la fois).
        Utilise G(y) = log cosh(a₁ y) / a₁ (plus robuste que kurtosis).
        """
        k, n = X_white.shape
        W = np.zeros((k, k))

        for i in range(k):
            w = np.random.randn(k)
            w /= np.linalg.norm(w)

            for _ in range(self.max_iter):
                u = w @ X_white                              # (n,)
                g_u = np.tanh(self.A1 * u)                  # g(y)
                g_prime = self.A1 * (1.0 - g_u ** 2)       # g'(y)

                w_new = (X_white @ g_u) / n - g_prime.mean() * w

                # Décorrélation de Gram-Schmidt
                if i > 0:
                    w_new -= W[:i].T @ (W[:i] @ w_new)

                norm = np.linalg.norm(w_new)
                if norm < 1e-12:
                    break
                w_new /= norm

                # Convergence
                if abs(abs(w_new @ w) - 1.0) < self.tol:
                    w = w_new
                    break
                w = w_new

            W[i] = w

        return W   # (k, k) matrice de séparation dans l'espace blanchi

    def _negentropy(self, y: np.ndarray) -> float:
        """
        Négentropie approximée : J(y) ≈ [E{G(y)} − E{G(ν)}]²
        avec G(y) = log cosh(y),  ν ~ N(0,1)
        Gaussien → J=0, super-gaussien → J > 0 (bon signal satellite LEO)
        """
        Ey = np.mean(np.log(np.cosh(self.A1 * y))) / self.A1
        # Pour ν ~ N(0,1) : E{log cosh(ν)} ≈ 0.3746
        Egauss = 0.3746
        return float((Ey - Egauss) ** 2)

    def separate(
        self,
        X: np.ndarray,
        doppler_profile: Optional[np.ndarray] = None,
        t_ms: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Sépare les sources et sélectionne la source satellite.
        Retourne (signal_satellite_isolé, index_source).
        """
        # ── 1. Blanchiment SVD ────────────────────────────────────────────
        X_white, W_white = self._svd_whitening(X)
        self.W_white = W_white

        # ── 2. FastICA dégonflage ────────────────────────────────────────
        W_sep = self._fastICA_deflation(X_white)
        W_full = W_sep @ W_white          # Matrice de séparation complète
        self.W = W_full

        k = W_sep.shape[0]
        S = W_sep @ X_white               # Sources séparées (k, n)

        # ── 3. Score synaptique par source ────────────────────────────────
        scores = np.zeros(k)
        n_samp = S.shape[1]

        for i in range(k):
            src = S[i]

            # Score négentropie : favorise les sources super-gaussiennes
            neg = self._negentropy(src)

            # Corrélation avec le profil Doppler attendu
            if doppler_profile is not None and len(doppler_profile) >= n_samp:
                dp = doppler_profile[:n_samp]
                norm_a = np.linalg.norm(np.abs(src)) + 1e-10
                norm_b = np.linalg.norm(np.abs(dp)) + 1e-10
                corr = float(np.abs(np.abs(src) @ np.abs(dp)) / (norm_a * norm_b))
            else:
                corr = 0.0

            # Score modulé par LTP et inhibition synaptique
            raw = neg * 0.5 + corr * 0.3 + self.ltp_traces[i] * 0.2
            mod = 1.0 + self.excitation[i] - self.inhibition[i]
            scores[i] = max(0.0, raw * mod)

        # ── 4. Sélection WTA + inhibition latérale ────────────────────────
        winner = int(np.argmax(scores))
        s_max  = scores[winner]

        for i in range(k):
            t_now = t_ms + float(i) * 0.1   # décalage temporel fictif pour STDP

            if i == winner:
                # LTP sur le gagnant
                self.source_weights[i] = self.stdp_states[i].post_spike(
                    t_now, self.source_weights[i]
                )
                self.ltp_traces[i]  = min(1.0, self.ltp_traces[i] + 0.15)
                self.excitation[i]  = min(1.0, self.excitation[i] + 0.08)
                self.inhibition[i] *= math.exp(-0.1 / BP.TAU_MINUS_MS)
            else:
                # LTD latéral proportionnel au score relatif
                overlap = scores[i] / max(s_max, 1e-9)
                self.source_weights[i] = self.stdp_states[i].pre_spike(
                    t_now, self.source_weights[i]
                )
                self.inhibition[i] = min(1.0, self.inhibition[i] + 0.3 * overlap)
                self.ltp_traces[i] *= math.exp(-1.0 / BP.TAU_LTP_MS)
                self.excitation[i] *= math.exp(-1.0 / BP.TAU_X_MS)

        return S[winner], winner


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 3 — SynapticMUSIC v2 : covariance FB-averaged + noyau DoG latéral
# ─────────────────────────────────────────────────────────────────────────────

class SynapticMUSIC:
    """
    MUSIC avec covariance forward-backward et inhibition latérale DoG.

    Covariance FB-averaged (améliore les performances en milieu cohérent) :
        R_fb = (R_f + J · R_f* · J) / 2
        R_f  = X X^H / N
        J    = matrice d'échange (anti-diagonale de 1)

    Spectre MUSIC :
        P(θ) = 1 / (a^H(θ) · E_n · E_n^H · a(θ))

    Vecteur directeur (ULA, λ/2 inter-élément) :
        a_k(θ) = exp(j·π·k·sin(θ)),   k = 0, …, M−1

    Inhibition latérale (Difference of Gaussians, DoG) :
        w(d) = A_exc · exp(−d²/2σ_exc²) − A_inh · exp(−d²/2σ_inh²)
    Le spectre MUSIC est convolutionné avec ce noyau pour supprimer
    les pics secondaires (multipath).

    Mise à jour LTP des directions :
        LTP(θ,t+1) = LTP(θ,t) · exp(−dt/τ_ltp) + Δ(θ ≈ θ*)
    """

    DOG_A_EXC  = 1.5     # Amplitude gaussienne centrale (excitation)
    DOG_A_INH  = 0.7     # Amplitude gaussienne latérale (inhibition)
    DOG_S_EXC  = 2.0     # Écart-type excitation (degrés)
    DOG_S_INH  = 6.0     # Écart-type inhibition (degrés)

    def __init__(self, n_elements: int = 8, n_sources: int = 1,
                 angle_range: Tuple[float, float] = (-90.0, 90.0),
                 angle_step_deg: float = 0.5):
        self.M = n_elements
        self.d = n_sources
        self.angle_grid = np.arange(angle_range[0], angle_range[1] + angle_step_deg, angle_step_deg)
        self.K = len(self.angle_grid)
        self.step = angle_step_deg

        # Kernel DoG précalculé
        self._dog_kernel = self._build_dog_kernel()

        # LTP weights sur les directions (mémoire longue durée)
        self.ltp_weights = np.ones(self.K)
        self.ltp_traces  = np.zeros(self.K)

        # Matrice J d'échange
        self._J = np.eye(self.M)[::-1]

        self.stdp = STDPState()
        self.w_direction: float = 0.5
        self._last_peak_idx: int = self.K // 2

    def _build_dog_kernel(self) -> np.ndarray:
        """Construit le noyau DoG sur la grille angulaire."""
        d = np.arange(-(self.K // 2), self.K // 2 + 1) * self.step
        exc = self.DOG_A_EXC * np.exp(-d ** 2 / (2 * self.DOG_S_EXC ** 2))
        inh = self.DOG_A_INH * np.exp(-d ** 2 / (2 * self.DOG_S_INH ** 2))
        kernel = exc - inh
        # Normalisation : somme positive = 1
        pos = kernel[kernel > 0]
        if pos.sum() > 0:
            kernel /= pos.sum()
        return kernel

    def _steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        a(θ) = [exp(j·π·0·sin θ), exp(j·π·1·sin θ), …, exp(j·π·(M-1)·sin θ)]
        ULA avec d = λ/2 → φ_k = π·k·sin(θ)
        """
        sin_t = math.sin(math.radians(theta_deg))
        k_vec = np.arange(self.M, dtype=float)
        return np.exp(1j * math.pi * k_vec * sin_t)

    def _fb_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Covariance forward-backward :
            R_f  = X X^H / N
            R_fb = (R_f + J R_f* J) / 2

        Avantage : découle les sources cohérentes (multipath),
        améliore le rang effectif de la matrice.
        """
        N = X.shape[1]
        R_f  = (X @ X.conj().T) / N
        J    = self._J
        R_fb = 0.5 * (R_f + J @ R_f.conj() @ J)
        # Régularisation diagonale pour stabilité numérique
        R_fb += np.eye(self.M) * 1e-9 * np.trace(R_fb) / self.M
        return R_fb

    def estimate_aoa(self, X: np.ndarray, t_ms: float = 0.0) -> AOAEstimate:
        """
        Estimation AOA via MUSIC FB-averaged + post-traitement DoG + LTP.
        X : (M, N) complexe
        """
        # ── 1. Covariance FB ──────────────────────────────────────────────
        R_fb = self._fb_covariance(X)

        # ── 2. Décomposition valeurs propres ─────────────────────────────
        eigvals, eigvecs = np.linalg.eigh(R_fb)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Sous-espace bruit
        En = eigvecs[:, self.d:]   # (M, M-d)

        # Projecteur bruit : Π_n = En En^H
        Pi_n = En @ En.conj().T

        # ── 3. Spectre MUSIC ──────────────────────────────────────────────
        spectrum = np.zeros(self.K)
        for k, theta in enumerate(self.angle_grid):
            a = self._steering_vector(theta)
            denom = np.real(a.conj() @ Pi_n @ a)
            spectrum[k] = 1.0 / max(denom, 1e-12)

        spectrum /= np.max(spectrum)

        # ── 4. Inhibition latérale DoG ────────────────────────────────────
        # Convolution avec le noyau DoG (mode "same" en périodique)
        dog_filtered = np.convolve(spectrum, self._dog_kernel, mode='same')
        dog_filtered = np.maximum(0.0, dog_filtered)
        if dog_filtered.max() > 0:
            dog_filtered /= dog_filtered.max()

        # ── 5. Modulation LTP ─────────────────────────────────────────────
        modulated = dog_filtered * self.ltp_weights
        peak_idx  = int(np.argmax(modulated))
        peak_val  = float(modulated[peak_idx])
        theta_hat = float(self.angle_grid[peak_idx])

        # ── 6. Mise à jour LTP STDP ───────────────────────────────────────
        # Fenêtre de renforcement : ±2° autour du pic
        half_win = int(2.0 / self.step)
        decay = math.exp(-1.0 / BP.TAU_LTP_MS)

        for k in range(self.K):
            self.ltp_weights[k] *= decay
            d_idx = abs(k - peak_idx)

            if d_idx <= half_win:
                # Zone gagnante : LTP + STDP
                gauss_reinf = math.exp(-(d_idx * self.step) ** 2 / (2 * self.DOG_S_EXC ** 2))
                self.ltp_traces[k] = min(1.0, self.ltp_traces[k] + 0.1 * gauss_reinf)
                self.ltp_weights[k] += BP.A_PLUS * self.ltp_traces[k]
                self.ltp_weights[k] = max(0.5, min(3.0, self.ltp_weights[k]))
            else:
                # Zone perdante : LTD latéral
                self.ltp_traces[k] *= decay

        self._last_peak_idx = peak_idx

        # Confiance : SNR du pic sur le spectre filtré
        mean_bg = np.mean(dog_filtered)
        confidence = min(1.0, peak_val / (mean_bg * 10.0 + 1e-9))

        # Élévation via géométrie LEO
        el_deg = self._elevation_from_azimuth(theta_hat)

        return AOAEstimate(
            azimuth_deg=theta_hat,
            elevation_deg=el_deg,
            confidence=confidence,
            music_peak=peak_val,
            timestamp_ms=t_ms,
        )

    def _elevation_from_azimuth(self, az_deg: float) -> float:
        """
        Estimation d'élévation par la géométrie sphérique (triangle de terre).
        el = arcsin(h / d_slant)
        d_slant = sqrt(h² + (R_e · sin(|az|))²)
        """
        h   = BP.LEO_ALT_KM
        R_e = BP.EARTH_R_KM
        lateral = R_e * math.sin(math.radians(min(abs(az_deg), 89.9)))
        d_slant = math.sqrt(h ** 2 + lateral ** 2)
        return math.degrees(math.asin(h / d_slant))


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 4 — SynapticIMM v2 : IMM bayésien complet + STDP sur poids de mélange
# ─────────────────────────────────────────────────────────────────────────────

class SynapticIMM:
    """
    IMM bayésien complet (Bar-Shalom 2002) avec plasticité synaptique.

    ── Formulation IMM complète ──────────────────────────────────────────────

    1. Mélange d'interaction :
        μ_j|i(k-1) = p_ij · μ_i(k-1) / c_j
        c_j = Σ_i p_ij · μ_i(k-1)     (normalisation)

    2. Prédiction par modèle :
        xᵢ⁰(k-1) = Σ_j μ_j|i(k-1) · x̂_j(k-1)
        Pᵢ⁰(k-1) = Σ_j μ_j|i (Pⱼ + (x̂ⱼ−xᵢ⁰)(x̂ⱼ−xᵢ⁰)ᵀ)
        x̂ᵢ(k|k-1) = Fᵢ · xᵢ⁰(k-1)
        Pᵢ(k|k-1)  = Fᵢ Pᵢ⁰ Fᵢᵀ + Qᵢ

    3. Mise à jour Kalman :
        Sᵢ = Hᵢ Pᵢ(k|k-1) Hᵢᵀ + Rᵢ
        Kᵢ = Pᵢ(k|k-1) Hᵢᵀ Sᵢ⁻¹
        x̂ᵢ(k) = x̂ᵢ(k|k-1) + Kᵢ νᵢ
        Pᵢ(k)  = (I − Kᵢ Hᵢ) Pᵢ(k|k-1)

    4. Vraisemblance exacte (Gaussienne) :
        Λᵢ = N(νᵢ ; 0, Sᵢ)
            = det(2π Sᵢ)^{-1/2} · exp(−νᵢᵀ Sᵢ⁻¹ νᵢ / 2)

    5. Mise à jour probabilités :
        μᵢ(k) = Λᵢ · cᵢ / (Σⱼ Λⱼ · cⱼ)

    ── Ajout STDP ────────────────────────────────────────────────────────────
    Le poids synaptique w_i module μ_i avant la normalisation :
        μ̃ᵢ = μᵢ · w_i       puis renormalisation
    Après mise à jour :
        - Modèle gagnant (Λᵢ max) : STDP LTP  → w_i augmente
        - Modèles perdants         : STDP LTD  → w_i diminue
    """

    MODELS = ["static", "walk", "run", "erratic"]

    # Matrices de transition Markov (somme par ligne = 1)
    P_TRANS = np.array([
        [0.80, 0.10, 0.07, 0.03],
        [0.05, 0.80, 0.12, 0.03],
        [0.02, 0.10, 0.83, 0.05],
        [0.02, 0.05, 0.08, 0.85],
    ])

    # Bruits de processus Q (variance position/vitesse)
    Q_SCALES = {"static": 0.01, "walk": 0.5, "run": 2.0, "erratic": 20.0}

    def __init__(self, dim_x: int = 6, dim_z: int = 3):
        """
        dim_x = 6 : état [x, y, z, vx, vy, vz]
        dim_z = 3 : mesure [x, y, z]
        """
        self.n = len(self.MODELS)
        self.dx = dim_x
        self.dz = dim_z

        # État initial par modèle
        self.x_hat = {m: np.zeros(dim_x)          for m in self.MODELS}
        self.P     = {m: np.eye(dim_x) * 1000.0   for m in self.MODELS}

        # Probabilités initiales uniformes
        self.mu = np.ones(self.n) / self.n

        # Poids synaptiques STDP
        self.w_syn   = np.ones(self.n) * 0.5
        self.stdp_st = [STDPState() for _ in range(self.n)]
        self.ltp_traces = np.zeros(self.n)

        # Matrice d'observation H : on observe les 3 premières composantes
        self.H = np.zeros((dim_z, dim_x))
        self.H[:dim_z, :dim_z] = np.eye(dim_z)

        # Bruit de mesure R
        self.R = np.eye(dim_z) * 25.0    # σ_meas = 5 km

        self.timestamp_ms: float = 0.0

    def _F_matrix(self, dt_s: float) -> np.ndarray:
        """Matrice de transition (mouvement uniformément accéléré)."""
        F = np.eye(self.dx)
        F[:3, 3:] = np.eye(3) * dt_s
        return F

    def _Q_matrix(self, model: str, dt_s: float) -> np.ndarray:
        """Bruit de processus discrétisé (Singer 1970)."""
        q = self.Q_SCALES[model]
        # Modèle de Singer : corrélation en dt²/2 et dt
        Q = np.zeros((self.dx, self.dx))
        Q[:3, :3] = np.eye(3) * q * (dt_s ** 3) / 3.0
        Q[3:, 3:] = np.eye(3) * q * dt_s
        Q[:3, 3:] = np.eye(3) * q * (dt_s ** 2) / 2.0
        Q[3:, :3] = Q[:3, 3:].T
        return Q

    def _gaussian_likelihood(self, nu: np.ndarray, S: np.ndarray) -> float:
        """
        Vraisemblance Gaussienne N(ν; 0, S).
        Λ = (2π)^{-d/2} |S|^{-1/2} exp(-νᵀ S⁻¹ ν / 2)
        Calcul en log pour éviter les underflows.
        """
        d = len(nu)
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                return 1e-300
            S_inv = np.linalg.solve(S, np.eye(d))
            mahal2 = float(nu @ S_inv @ nu)
            log_L = -0.5 * (d * math.log(2 * math.pi) + logdet + mahal2)
            return math.exp(max(log_L, -700))   # Évite underflow
        except np.linalg.LinAlgError:
            return 1e-300

    def predict_update(
        self,
        z: Optional[np.ndarray],   # Mesure (3,) ou None si signal perdu
        dt_ms: float = 100.0,
    ) -> TrajectoryState:
        """
        Cycle IMM complet : interaction → prédiction → (update) → fusion.
        """
        self.timestamp_ms += dt_ms
        dt_s = dt_ms * 1e-3
        F    = self._F_matrix(dt_s)
        idx  = {m: i for i, m in enumerate(self.MODELS)}

        # ── 1. Calcul des probabilités de mélange μ_{j|i} ─────────────────
        # c_j = Σ_i p_ij μ_i
        c_bar = self.P_TRANS.T @ self.mu   # (n,)
        c_bar = np.maximum(c_bar, 1e-300)

        # μ_{j|i} = p_ij μ_i / c_j   (matrice n×n)
        mu_ji = (self.P_TRANS * self.mu[:, None]) / c_bar[None, :]   # (n, n)

        # ── 2. États mixés pour chaque modèle ──────────────────────────────
        x0 = {}
        P0 = {}
        for j, m_j in enumerate(self.MODELS):
            x0_j = np.zeros(self.dx)
            for i, m_i in enumerate(self.MODELS):
                x0_j += mu_ji[i, j] * self.x_hat[m_i]
            x0[m_j] = x0_j

            P0_j = np.zeros((self.dx, self.dx))
            for i, m_i in enumerate(self.MODELS):
                d_x = self.x_hat[m_i] - x0_j
                P0_j += mu_ji[i, j] * (self.P[m_i] + np.outer(d_x, d_x))
            P0[m_j] = P0_j

        # ── 3. Prédiction Kalman par modèle ───────────────────────────────
        x_pred, P_pred = {}, {}
        for m in self.MODELS:
            Q = self._Q_matrix(m, dt_s)
            x_pred[m] = F @ x0[m]
            P_pred[m] = F @ P0[m] @ F.T + Q

        # ── 4. Mise à jour Kalman (si mesure disponible) ─────────────────
        x_upd, P_upd, Lambda = {}, {}, {}
        for m in self.MODELS:
            if z is not None:
                nu = z - self.H @ x_pred[m]
                S  = self.H @ P_pred[m] @ self.H.T + self.R
                try:
                    K = np.linalg.solve(S.T, (P_pred[m] @ self.H.T).T).T
                except np.linalg.LinAlgError:
                    K = P_pred[m] @ self.H.T @ np.linalg.pinv(S)

                x_upd[m] = x_pred[m] + K @ nu
                P_upd[m] = (np.eye(self.dx) - K @ self.H) @ P_pred[m]
                Lambda[m] = self._gaussian_likelihood(nu, S)
            else:
                # Pas de mesure : pas d'update
                x_upd[m] = x_pred[m]
                P_upd[m] = P_pred[m]
                Lambda[m] = self.mu[idx[m]]   # Maintenir probabilité courante

        # ── 5. Mise à jour des probabilités modèles + STDP ────────────────
        # μ̃_i = Λ_i · c_bar_i · w_syn_i
        mu_new = np.array([Lambda[m] * c_bar[i] * self.w_syn[i] for i, m in enumerate(self.MODELS)])
        total  = mu_new.sum()
        if total < 1e-300:
            mu_new = np.ones(self.n) / self.n
        else:
            mu_new /= total
        self.mu = mu_new

        # STDP : renforcer le modèle gagnant
        winner_idx = int(np.argmax(self.mu))
        for i, m in enumerate(self.MODELS):
            t_now = self.timestamp_ms + i * 0.1
            if i == winner_idx:
                self.w_syn[i] = self.stdp_st[i].post_spike(t_now, self.w_syn[i])
                self.ltp_traces[i] = min(1.0, self.ltp_traces[i] + BP.A_PLUS)
            else:
                self.w_syn[i] = self.stdp_st[i].pre_spike(t_now, self.w_syn[i])
                self.ltp_traces[i] *= math.exp(-1.0 / BP.TAU_LTP_MS)

        # ── 6. Fusion globale ─────────────────────────────────────────────
        x_fused = np.zeros(self.dx)
        P_fused = np.zeros((self.dx, self.dx))
        for i, m in enumerate(self.MODELS):
            self.x_hat[m] = x_upd[m]
            self.P[m]     = P_upd[m]
            x_fused += self.mu[i] * x_upd[m]

        for i, m in enumerate(self.MODELS):
            d_x = x_upd[m] - x_fused
            P_fused += self.mu[i] * (P_upd[m] + np.outer(d_x, d_x))

        model_probs = {m: float(self.mu[i]) for i, m in enumerate(self.MODELS)}

        return TrajectoryState(
            position_km=x_fused[:3].copy(),
            velocity_kms=x_fused[3:].copy(),
            cov=P_fused,
            model_probs=model_probs,
            timestamp_ms=self.timestamp_ms,
            predicted=(z is None),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 5 — SynapticPINN v2 : RK4 + J2 + drag + STDP
# ─────────────────────────────────────────────────────────────────────────────

class SynapticPINN:
    """
    Physics-Informed Neural Network orbital avec :

    ── Intégration RK4 ────────────────────────────────────────────────────────
    k₁ = f(t,     y)
    k₂ = f(t+h/2, y + h/2 k₁)
    k₃ = f(t+h/2, y + h/2 k₂)
    k₄ = f(t+h,   y + h k₃)
    y(t+h) = y + h/6 (k₁ + 2k₂ + 2k₃ + k₄)

    ── Dynamique orbitale complète ────────────────────────────────────────────
    Gravité newtonienne :
        a_grav = −GM/r³ · r

    Perturbation J2 (aplatissement terrestre) :
        a_J2x = −3/2 · J2 · GM · Re² / r⁵ · x · (1 − 5z²/r²)
        a_J2z = −3/2 · J2 · GM · Re² / r⁵ · z · (3 − 5z²/r²)
        (idem y, similaire à x)

    Traînée atmosphérique :
        a_drag = −1/2 · Cd · (A/m) · ρ(r) · |v| · v

    ── Loss PINN ─────────────────────────────────────────────────────────────
        L = MSE(x_pred, x_true) + λ · ‖ẋ_pred − f(x_pred)‖²

    ── STDP sur les neurones cachés ──────────────────────────────────────────
    Les neurones dont les activations précèdent une correction (Δt > 0)
    sont renforcés via LTP.
    """

    def __init__(self, n_neurons: int = 64, lambda_physics: float = 0.1):
        self.n  = n_neurons
        self.lp = lambda_physics

        # Réseau à 1 couche cachée (dim_in=6, dim_hidden=n, dim_out=6)
        scale_in  = math.sqrt(2.0 / 6)
        scale_out = math.sqrt(2.0 / n_neurons)
        self.W1 = np.random.randn(n_neurons, 6)     * scale_in
        self.b1 = np.zeros(n_neurons)
        self.W2 = np.random.randn(6, n_neurons)     * scale_out
        self.b2 = np.zeros(6)

        # STDP par neurone
        self.stdp_neurons = [STDPState() for _ in range(n_neurons)]
        self.neuron_ltp   = np.zeros(n_neurons)
        self.last_fire_ms = np.full(n_neurons, -np.inf)

        self.history: List[Tuple[float, np.ndarray]] = []
        self.lr = 1e-4           # Taux d'apprentissage
        self._t_last_update: float = 0.0

    # ── Dynamique orbitale ────────────────────────────────────────────────────

    def _orbital_rhs(self, state: np.ndarray) -> np.ndarray:
        """
        Retourne ẋ = f(x) = [v, a_total].
        Inclut gravité newtonienne + J2 + traînée atmosphérique.
        """
        r_vec = state[:3]
        v_vec = state[3:]
        # Clamp pour stabilité numérique (évite r→0 ou r>>R_atm)
        r_mag_raw = np.linalg.norm(r_vec)
        if r_mag_raw < BP.EARTH_R_KM:
            # Re-centrer sur orbite nominale si drift excessif
            r_vec = r_vec / max(r_mag_raw, 1.0) * (BP.EARTH_R_KM + BP.LEO_ALT_KM)
        if r_mag_raw > 1e6:
            r_vec = r_vec / r_mag_raw * (BP.EARTH_R_KM + BP.LEO_ALT_KM)
        r     = np.linalg.norm(r_vec) + 1e-9
        r2    = r ** 2
        r3    = r ** 3
        r5    = r ** 5

        # ── Gravité newtonienne ────────────────────────────────────────────
        a_grav = -(BP.GM / r3) * r_vec

        # ── Perturbation J2 ────────────────────────────────────────────────
        Re = BP.EARTH_R_KM
        z  = r_vec[2]
        c  = -1.5 * BP.J2 * BP.GM * Re ** 2 / r5
        fac = 5.0 * z ** 2 / r2
        a_j2 = c * np.array([
            r_vec[0] * (1.0 - fac),
            r_vec[1] * (1.0 - fac),
            r_vec[2] * (3.0 - fac),
        ])

        # ── Traînée atmosphérique (modèle exponentiel simplifié) ──────────
        alt   = r - BP.EARTH_R_KM
        # Densité exponentielle : ρ(h) = ρ₀ · exp(−(h−h₀)/H_scale)
        H_scale = 60.0   # km
        rho = BP.RHO_500KM * math.exp(-(alt - 500.0) / H_scale)
        v_mag = np.linalg.norm(v_vec) + 1e-9
        a_drag = -0.5 * BP.CD * BP.A_OVER_M * rho * v_mag * v_vec * 1e6  # conversion km→m

        a_total = a_grav + a_j2 + a_drag
        return np.concatenate([v_vec, a_total])

    def _rk4(self, state: np.ndarray, dt_s: float) -> np.ndarray:
        """Intégration RK4 de la dynamique orbitale."""
        k1 = self._orbital_rhs(state)
        k2 = self._orbital_rhs(state + 0.5 * dt_s * k1)
        k3 = self._orbital_rhs(state + 0.5 * dt_s * k2)
        k4 = self._orbital_rhs(state + dt_s * k3)
        return state + (dt_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ── Réseau neuronal ───────────────────────────────────────────────────────

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Passe avant. Retourne (sortie, activations couche cachée)."""
        h = np.tanh(self.W1 @ x + self.b1)   # (n,)
        y = self.W2 @ h + self.b2             # (6,)
        return y, h

    def predict(self, state: np.ndarray, dt_s: float, t_ms: float) -> np.ndarray:
        """
        Prédiction = intégration RK4 + correction neurale.
        x_pred = RK4(x, dt) + α · NN(x)
        où α est petit (le réseau corrige, la physique domine).
        """
        x_phys = self._rk4(state, dt_s)

        correction, h = self._forward(state)
        alpha = 0.001   # La physique domine — correction neurale très faible
        x_pred = x_phys + alpha * correction

        # STDP : enregistrer les neurones qui ont "fire"
        fired = np.abs(h) > 0.5
        self.last_fire_ms[fired] = t_ms
        self.neuron_ltp[fired] = np.minimum(1.0, self.neuron_ltp[fired] + 0.05)
        self.neuron_ltp[~fired] *= math.exp(-1.0 / BP.TAU_LTP_MS)

        self.history.append((t_ms, x_pred.copy()))
        return x_pred

    def update(self, true_state: np.ndarray, t_ms: float) -> float:
        """
        Mise à jour du réseau et des poids STDP.
        Retourne la loss totale (MSE + terme physique).
        """
        if not self.history:
            return 0.0

        x_pred = self.history[-1][1]

        # ── Loss PINN ─────────────────────────────────────────────────────
        ts6 = np.zeros(6)
        n   = min(len(true_state), 6)
        ts6[:n] = true_state[:n]

        mse_loss    = np.mean((x_pred - ts6) ** 2)
        # Résidu physique : ẋ_pred − f(x_pred)
        x_dot_phys  = self._orbital_rhs(x_pred)
        correction, h = self._forward(x_pred)
        x_dot_pred  = x_dot_phys + 0.005 * correction
        phys_resid  = np.mean((x_dot_pred - x_dot_phys) ** 2)

        total_loss = mse_loss + self.lp * phys_resid

        # ── Gradient approx (1er ordre) ───────────────────────────────────
        err = x_pred - ts6
        dL_dy = 2.0 * err / 6.0

        # Backprop couche de sortie
        dL_dW2 = np.outer(dL_dy, h)
        dL_db2 = dL_dy

        # Backprop couche cachée
        dL_dh  = self.W2.T @ dL_dy
        dtanh  = 1.0 - h ** 2                  # dérivée tanh
        dL_dpre = dL_dh * dtanh

        dL_dW1 = np.outer(dL_dpre, x_pred)
        dL_db1 = dL_dpre

        # Mise à jour des poids (gradient descent)
        self.W2 -= self.lr * np.clip(dL_dW2, -1.0, 1.0)
        self.b2 -= self.lr * np.clip(dL_db2, -1.0, 1.0)
        self.W1 -= self.lr * np.clip(dL_dW1, -1.0, 1.0)
        self.b1 -= self.lr * np.clip(dL_db1, -1.0, 1.0)
        # Clamp les poids pour éviter l'explosion
        self.W1 = np.clip(self.W1, -3.0, 3.0)
        self.W2 = np.clip(self.W2, -3.0, 3.0)

        # ── STDP sur les neurones ─────────────────────────────────────────
        for i in range(self.n):
            dt_fire = t_ms - self.last_fire_ms[i]
            if 0.0 < dt_fire <= BP.TAU_PLUS_MS:
                # Neurone a fire avant la correction → LTP
                ltp_dw = BP.A_PLUS * math.exp(-dt_fire / BP.TAU_PLUS_MS)
                self.W2[:, i] += ltp_dw * dL_dy * self.neuron_ltp[i]
                self.W2[:, i] = np.clip(self.W2[:, i], -2.0, 2.0)

        self._t_last_update = t_ms
        return float(total_loss)


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 6 — MeshRouter v2 : consensus Gossip synaptique
# ─────────────────────────────────────────────────────────────────────────────

class Synapse:
    """Lien synaptique inter-station avec STDP paire."""

    def __init__(self, pre: str, post: str, dist_km: float):
        self.pre  = pre
        self.post = post
        self.dist = dist_km
        self.w    = random.uniform(0.3, 0.7)
        self.stdp = STDPState()
        self.g    = 0.0       # Conductance courante
        self.n_tx = 0
        self.n_ok = 0

    @property
    def delay_ms(self) -> float:
        return self.dist / BP.C_KM_PER_MS

    @property
    def w_eff(self) -> float:
        """Poids effectif (conductance modulée par poids plastique)."""
        return min(1.0, self.w * (1.0 + self.g))

    def transmit(self, pkt: SpikePacket, t_ms: float) -> Optional[SpikePacket]:
        """Transmet un paquet via la synapse. Retourne None si bloqué."""
        self.n_tx += 1
        self.g = max(0.0, self.g - 0.05)   # Décroissance conductance

        strength = pkt.mean_intensity * self.w_eff
        if strength >= 0.3:
            self.n_ok += 1
            # STDP : pré → w+
            self.w = self.stdp.pre_spike(t_ms, self.w)
            # Conductance synaptique += poids spike
            self.g = min(2.0, self.g + BP.G_MAX * self.w)

            return SpikePacket(
                origin_id=self.pre,
                timestamp_ms=t_ms + self.delay_ms,
                spike_times=[s + self.delay_ms for s in pkt.spike_times],
                intensities=[v * self.w_eff    for v in pkt.intensities],
                payload_bits=pkt.payload_bits,
                snr_db=pkt.snr_db,
            )
        return None

    def post_received(self, t_ms: float) -> None:
        """Appelé quand la station post reçoit et valide le paquet."""
        self.w = self.stdp.post_spike(t_ms, self.w)


class MeshRouter:
    """
    Consensus Gossip synaptique entre stations terrestres.

    Algorithme de consensus (Xiao & Boyd 2004) :
        x_i(t+1) = x_i(t) + ε · Σ_j w_ij · (x_j(t) − x_i(t))

    Ici w_ij = poids synaptique effectif de la synapse i→j (normalisé).
    ε = pas de consensus (0 < ε < 1 / λ_max(L))

    Convergence garantie si le graphe est connexe et ε < 1 / Δ_max.
    """

    EPSILON = 0.3   # Pas de consensus

    def __init__(self):
        self.stations: List[str]                     = []
        self.synapses: Dict[Tuple[str,str], Synapse] = {}

    def add_station(self, sid: str) -> None:
        if sid not in self.stations:
            self.stations.append(sid)

    def add_link(self, a: str, b: str, dist_km: float, bidir: bool = True) -> None:
        self.synapses[(a, b)] = Synapse(a, b, dist_km)
        if bidir:
            self.synapses[(b, a)] = Synapse(b, a, dist_km)

    def gossip_consensus(
        self,
        estimates: Dict[str, AOAEstimate],
        n_rounds: int = 3,
    ) -> AOAEstimate:
        """
        Consensus Gossip pondéré par LTP sur n_rounds itérations.

        Vecteurs d'état : x_i = [az_i, el_i, conf_i]
        w_ij = w_eff de la synapse i→j, normalisé par degré sortant.
        """
        if not estimates:
            raise ValueError("Aucune estimation disponible")
        if len(estimates) == 1:
            return next(iter(estimates.values()))

        # Initialiser les états
        state: Dict[str, np.ndarray] = {}
        for sid, aoa in estimates.items():
            state[sid] = np.array([aoa.azimuth_deg, aoa.elevation_deg, aoa.confidence])

        # Gossip rounds
        for _ in range(n_rounds):
            new_state = {sid: s.copy() for sid, s in state.items()}
            for sid in estimates:
                # Voisins sortants
                neighbors = [(post, syn) for (pre, post), syn in self.synapses.items()
                             if pre == sid and post in estimates]
                if not neighbors:
                    continue
                # Normalisation des poids
                w_sum = sum(syn.w_eff for _, syn in neighbors)
                if w_sum < 1e-9:
                    continue
                for post, syn in neighbors:
                    w_norm = syn.w_eff / w_sum
                    new_state[sid] += self.EPSILON * w_norm * (state[post] - state[sid])
            state = new_state

        # Fusion finale : moyenne pondérée par confiance finale
        totals = np.zeros(3)
        w_total = 0.0
        for sid, s in state.items():
            w = max(s[2], 1e-6)   # Confiance comme poids
            totals += w * s
            w_total += w
        final = totals / max(w_total, 1e-9)

        return AOAEstimate(
            azimuth_deg=float(final[0]),
            elevation_deg=float(final[1]),
            confidence=min(1.0, float(final[2])),
            music_peak=1.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOC 7 — SynapticAnomalyDetector v2 : Mahalanobis régularisé + KL sur ISI
# ─────────────────────────────────────────────────────────────────────────────

class SynapticAnomalyDetector:
    """
    Double détection de spoofing :

    ── Test 1 : Mahalanobis régularisé (Ledoit-Wolf) ─────────────────────────
        D² = (x − μ)ᵀ Σ_reg⁻¹ (x − μ)
        Σ_reg = (1 − α) · Σ̂ + α · I
        α = Ledoit-Wolf shrinkage estimator

    ── Test 2 : Divergence KL sur les ISI ────────────────────────────────────
    Le train de spikes d'un vrai signal LEO a une distribution ISI
    caractéristique (approximativement inverse-Gaussienne).
    Un signal spoofé a une distribution ISI anormale.

        D_KL(P_obs ‖ P_ltp) = Σ_k P_obs(k) · log(P_obs(k) / P_ltp(k))

    P_ltp = distribution ISI apprise via LTP sur l'historique.
    P_obs = distribution ISI du paquet courant.

    ── Décision ─────────────────────────────────────────────────────────────
        Alerte si D² > χ²_α(d)  OU  D_KL > seuil_KL
    """

    # Seuil χ² à d=7, α=0.01 → χ²(7, 0.01) ≈ 18.5
    CHI2_THRESHOLD = 18.5
    KL_THRESHOLD   = 0.5
    N_ISI_BINS     = 20
    ALPHA_SHRINK   = 0.1   # Régularisation Ledoit-Wolf

    def __init__(self, history_len: int = 100):
        self.history_len = history_len
        self.feature_hist: List[np.ndarray] = []

        # Distribution ISI apprise (LTP)
        self.isi_ltp = np.ones(self.N_ISI_BINS) / self.N_ISI_BINS
        self.isi_ltp_count: int = 0
        self.n_alerts: int = 0

    def _extract_features(self, aoa: AOAEstimate, pkt: SpikePacket) -> np.ndarray:
        """
        Vecteur de features pour Mahalanobis (d=7).
        Toutes normalisées pour éviter les problèmes de conditionnement.
        """
        n_spk  = max(1, len(pkt.spike_times))
        rate   = pkt.firing_rate_hz
        m_int  = pkt.mean_intensity
        isi_cv = 0.0
        if len(pkt.isi) > 1:
            isi_mean = np.mean(pkt.isi)
            isi_std  = np.std(pkt.isi)
            isi_cv   = isi_std / max(isi_mean, 1e-6)

        return np.array([
            aoa.azimuth_deg / 90.0,          # Normalisé [-1, 1]
            aoa.elevation_deg / 90.0,
            aoa.confidence,
            aoa.music_peak,
            m_int,
            min(rate / 100.0, 1.0),           # Normalisé
            min(isi_cv, 5.0) / 5.0,           # Coefficient de variation ISI
        ])

    def _ledoit_wolf_cov(self, H: np.ndarray) -> np.ndarray:
        """
        Estimateur de covariance Ledoit-Wolf régularisé :
            Σ_reg = (1 − α) · Σ̂ + α · I
        α = paramètre de shrinkage fixé (peut être estimé analytiquement).
        """
        n, d = H.shape
        mu   = H.mean(axis=0)
        Hc   = H - mu
        S    = (Hc.T @ Hc) / (n - 1)
        mu_S = np.trace(S) / d           # Cible sphérique : μ · I
        S_reg = (1.0 - self.ALPHA_SHRINK) * S + self.ALPHA_SHRINK * mu_S * np.eye(d)
        return S_reg

    def _mahalanobis(self, x: np.ndarray, H: np.ndarray) -> float:
        """
        D² = (x − μ)ᵀ Σ_reg⁻¹ (x − μ) avec covariance Ledoit-Wolf.
        """
        mu    = H.mean(axis=0)
        S_reg = self._ledoit_wolf_cov(H)
        try:
            S_inv = np.linalg.solve(S_reg, np.eye(len(x)))
            d_vec = x - mu
            return float(d_vec @ S_inv @ d_vec)
        except np.linalg.LinAlgError:
            return 0.0

    def _kl_isi(self, pkt: SpikePacket) -> float:
        """
        Divergence KL entre la distribution ISI observée et la distribution
        ISI apprise (LTP).

        D_KL(P_obs ‖ P_ltp) = Σ_k P_obs(k) · log(P_obs(k) / P_ltp(k))
        """
        isi = pkt.isi
        if len(isi) < 3:
            return 0.0

        # Histogramme ISI (bins logarithmiques pour couvrir 0.1–1000 ms)
        bins = np.logspace(-1, 3, self.N_ISI_BINS + 1)
        p_obs, _ = np.histogram(isi, bins=bins, density=False)
        p_obs    = p_obs.astype(float) + 1e-9
        p_obs   /= p_obs.sum()

        p_ltp = self.isi_ltp + 1e-9
        p_ltp /= p_ltp.sum()

        # D_KL(P_obs ‖ P_ltp) — seules les bins non nulles
        kl = float(np.sum(p_obs * np.log(p_obs / p_ltp)))
        return max(0.0, kl)

    def _update_isi_ltp(self, pkt: SpikePacket) -> None:
        """Mise à jour LTP de la distribution ISI de référence."""
        isi = pkt.isi
        if len(isi) < 3:
            return
        bins = np.logspace(-1, 3, self.N_ISI_BINS + 1)
        p_new, _ = np.histogram(isi, bins=bins, density=False)
        p_new = p_new.astype(float) + 1e-9
        p_new /= p_new.sum()

        # Mise à jour exponentielle (LTP)
        alpha = 1.0 / max(1.0, self.isi_ltp_count)
        alpha = min(alpha, 0.1)
        self.isi_ltp = (1.0 - alpha) * self.isi_ltp + alpha * p_new
        self.isi_ltp_count += 1

    def detect(
        self,
        aoa: AOAEstimate,
        pkt: SpikePacket,
    ) -> Tuple[bool, float, float]:
        """
        Retourne (is_spoofed, D²_mahalanobis, D_KL_ISI).
        """
        feat = self._extract_features(aoa, pkt)

        # ── Test 1 : Mahalanobis ─────────────────────────────────────────
        d2 = 0.0
        if len(self.feature_hist) >= 10:
            H  = np.array(self.feature_hist[-self.history_len:])
            d2 = self._mahalanobis(feat, H)

        # ── Test 2 : Divergence KL ISI ────────────────────────────────────
        kl = self._kl_isi(pkt)

        # ── Décision ─────────────────────────────────────────────────────
        mahal_alert = d2 > self.CHI2_THRESHOLD
        kl_alert    = kl > self.KL_THRESHOLD and self.isi_ltp_count > 5

        spoofed = mahal_alert or kl_alert

        if not spoofed:
            # Signal légitime → mettre à jour l'historique et le LTP ISI
            self.feature_hist.append(feat.copy())
            self._update_isi_ltp(pkt)

        if spoofed:
            self.n_alerts += 1

        return spoofed, d2, kl


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet — SynapticLEOPipeline v2
# ─────────────────────────────────────────────────────────────────────────────

class SynapticLEOPipeline:
    """
    Orchestre les 7 blocs synaptiques sur chaque trame RF entrante.
    """

    def __init__(self, station_id: str = "STA-1"):
        self.sid      = station_id
        self.encoder  = SpikeEncoder(station_id)
        self.ica      = SynapticICA(n_components=4)
        self.music    = SynapticMUSIC(n_elements=8, n_sources=1)
        self.imm      = SynapticIMM()
        self.pinn     = SynapticPINN(n_neurons=64)
        self.anomaly  = SynapticAnomalyDetector()
        # État initial : orbite circulaire LEO réaliste (plan équatorial)
        # r = R_earth + h = 6921 km, v_circ = sqrt(GM/r) ≈ 7.6 km/s
        r0 = BP.EARTH_R_KM + BP.LEO_ALT_KM
        v0 = math.sqrt(BP.GM / r0)
        self._last_state = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
        self._t_ms: float = 0.0

    def process(self, signal: RFSignal) -> Dict:
        self._t_ms = signal.timestamp_ms
        res: Dict = {"station": self.sid, "t_ms": self._t_ms}

        # ── Bloc 1 : LIF encoder ──────────────────────────────────────────
        pkt = self.encoder.encode(signal)
        res["n_spikes"]    = len(pkt.spike_times)
        res["compression"] = self.encoder.compression_ratio
        res["energy_J"]    = self.encoder.total_energy

        # Signal perdu → PINN prédit
        if not pkt.spike_times:
            x_pred = self.pinn.predict(self._last_state, dt_s=0.1, t_ms=self._t_ms)
            traj   = self.imm.predict_update(None, dt_ms=100.0)
            res["signal_lost"] = True
            res["pinn_pos_km"] = x_pred[:3].tolist()
            res["imm_probs"]   = traj.model_probs
            return res

        # ── Bloc 2 : SynapticICA ─────────────────────────────────────────
        n_samp = min(len(signal.iq_samples), 256)
        iq     = signal.iq_samples[:n_samp]

        # Matrice de mélange simulée (4 sources : satellite + 3 interférences)
        X = np.zeros((4, n_samp))
        X[0] = np.real(iq)
        X[1] = np.imag(iq)
        X[2] = np.random.randn(n_samp) * 0.2  # interférence 1
        X[3] = np.random.randn(n_samp) * 0.1  # interférence 2

        f_d = 7.8 / (BP.C_KM_PER_MS * 1e3) * BP.KU_FREQ_GHZ * 1e9
        t_v = np.linspace(0, 1e-3, n_samp)
        dop_profile = np.abs(iq) * np.cos(2 * np.pi * f_d * t_v)

        sat_sig, src_idx = self.ica.separate(X, dop_profile, self._t_ms)
        res["ica_src"]  = int(src_idx)
        res["ica_exc"]  = float(self.ica.excitation[src_idx])
        res["ica_ltp"]  = float(self.ica.ltp_traces[src_idx])

        # ── Bloc 3 : SynapticMUSIC ────────────────────────────────────────
        M = self.music.M
        X_ula = np.zeros((M, n_samp), dtype=complex)
        for k in range(M):
            phase = np.exp(1j * math.pi * k * math.sin(math.radians(30.0)))
            X_ula[k] = iq * phase

        aoa = self.music.estimate_aoa(X_ula, self._t_ms)
        res["aoa"] = {
            "az_deg":  round(aoa.azimuth_deg, 3),
            "el_deg":  round(aoa.elevation_deg, 3),
            "conf":    round(aoa.confidence, 4),
        }

        # ── Bloc 4 : SynapticIMM ─────────────────────────────────────────
        R_e  = BP.EARTH_R_KM
        h    = BP.LEO_ALT_KM
        az_r = math.radians(aoa.azimuth_deg)
        el_r = math.radians(aoa.elevation_deg)
        meas = np.array([
            h * math.cos(el_r) * math.cos(az_r),
            h * math.cos(el_r) * math.sin(az_r),
            h * math.sin(el_r),
        ])
        traj = self.imm.predict_update(meas, dt_ms=100.0)

        best_m  = max(traj.model_probs, key=lambda m: traj.model_probs[m])
        best_mu = traj.model_probs[best_m]
        best_w  = self.imm.w_syn[self.imm.MODELS.index(best_m)]
        res["imm"] = {
            "model": best_m,
            "prob":  round(best_mu, 4),
            "w_syn": round(float(best_w), 4),
            "ltp":   round(float(self.imm.ltp_traces[self.imm.MODELS.index(best_m)]), 4),
        }

        # ── Bloc 5 : SynapticPINN ────────────────────────────────────────
        self._last_state[:3] = meas
        self._last_state[3:] = traj.velocity_kms

        x_pred   = self.pinn.predict(self._last_state, dt_s=0.1, t_ms=self._t_ms)
        pinn_loss = self.pinn.update(meas, self._t_ms)
        drift_km = float(np.linalg.norm(x_pred[:3] - meas))
        res["pinn"] = {"drift_km": round(drift_km, 4), "loss": round(pinn_loss, 6)}

        # ── Bloc 7 : SynapticAnomalyDetector ─────────────────────────────
        spoofed, d2, kl = self.anomaly.detect(aoa, pkt)
        res["spoofing"] = {
            "alert":   spoofed,
            "D2":      round(d2, 3),
            "KL_ISI":  round(kl, 4),
            "n_alerts": self.anomaly.n_alerts,
        }

        return res


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def make_signal(snr_db: float, t_ms: float, spoof: bool = False) -> RFSignal:
    n = 512
    sr = 512_000.0
    fc = BP.KU_FREQ_GHZ * 1e9

    f_d   = 7.8 / (BP.C_KM_PER_MS * 1e3) * fc
    t_vec = np.linspace(0, n / sr, n)

    pwr = 10.0 ** (snr_db / 20.0)
    if spoof:
        # Doppler hors plage LEO + modulation anormale
        f_fake = f_d * 5.0
        src = pwr * np.exp(1j * (2*np.pi*f_fake*t_vec + np.cumsum(np.random.randn(n)*0.5)))
    else:
        src = pwr * np.exp(1j * (2*np.pi*f_d*t_vec + np.random.uniform(0, 2*np.pi)))

    multipath = 0.3*pwr * np.exp(1j * (2*np.pi*f_d*1.15*t_vec + np.random.uniform(0, 2*np.pi)))
    noise = (np.random.randn(n) + 1j*np.random.randn(n)) * 0.5
    iq = src + multipath + noise

    return RFSignal(iq_samples=iq, sample_rate_hz=sr, center_freq_hz=fc, snr_db=snr_db, timestamp_ms=t_ms)


def _fmt(r: Dict) -> None:
    if r.get("signal_lost"):
        p = [round(v, 1) for v in r.get("pinn_pos_km", [])]
        print(f"  ⚠ Signal perdu  | PINN pos : {p} km")
        return
    aoa = r.get("aoa", {})
    imm = r.get("imm", {})
    pinn = r.get("pinn", {})
    sp  = r.get("spoofing", {})
    status = "⚠ ALERTE" if sp.get("alert") else "✓ OK    "
    print(
        f"  Spikes:{r['n_spikes']:3d}  Compr:{r['compression']:.1%}  "
        f"Az:{aoa.get('az_deg',0):7.2f}°  El:{aoa.get('el_deg',0):5.2f}°  "
        f"Conf:{aoa.get('conf',0):.2%}  "
        f"IMM:{imm.get('model','?'):8s}(μ={imm.get('prob',0):.2f} w={imm.get('w_syn',0):.3f})  "
        f"PINN:{pinn.get('drift_km',0):.2f}km  "
        f"Spoof:{status} D²={sp.get('D2',0):.1f} KL={sp.get('KL_ISI',0):.3f}"
    )


def main():
    np.random.seed(42)
    print("=" * 80)
    print("  SynapticLEO v2 — Pipeline complet (formulation mathématique rigoureuse)")
    print("=" * 80)

    pipe = SynapticLEOPipeline("STA-BUJUMBURA")

    # ── Phase d'apprentissage (50 trames) ─────────────────────────────────
    print(f"\n▶ Apprentissage (50 trames, SNR=15 dB) ...")
    for i in range(50):
        pipe.process(make_signal(snr_db=15.0, t_ms=float(i * 100)))
    print("  Terminé.")

    # ── Trames nominales ──────────────────────────────────────────────────
    print(f"\n▶ Trames nominales (SNR=15 dB)")
    print("  " + "─"*76)
    for i in range(3):
        r = pipe.process(make_signal(snr_db=15.0, t_ms=5000.0 + i*100))
        _fmt(r)

    # ── Faible SNR ────────────────────────────────────────────────────────
    print(f"\n▶ Faible SNR (−5 dB)")
    print("  " + "─"*76)
    for i in range(3):
        r = pipe.process(make_signal(snr_db=-5.0, t_ms=6000.0 + i*100))
        _fmt(r)

    # ── Shadowing (signal perdu 5 trames) ─────────────────────────────────
    print(f"\n▶ Shadowing (5 trames sans signal)")
    print("  " + "─"*76)
    empty_sig = RFSignal(
        iq_samples=np.zeros(512, dtype=complex), sample_rate_hz=512000,
        center_freq_hz=BP.KU_FREQ_GHZ*1e9, snr_db=-30.0, timestamp_ms=7000.0
    )
    for i in range(5):
        empty_sig.timestamp_ms = 7000.0 + i * 100
        r = pipe.process(empty_sig)
        _fmt(r)

    # ── Injection spoofing ────────────────────────────────────────────────
    print(f"\n▶ Injection spoofing (5 trames)")
    print("  " + "─"*76)
    for i in range(5):
        r = pipe.process(make_signal(snr_db=20.0, t_ms=8000.0+i*100, spoof=True))
        _fmt(r)

    # ── Récupération post-spoofing ─────────────────────────────────────────
    print(f"\n▶ Récupération post-spoofing (SNR=15 dB)")
    print("  " + "─"*76)
    for i in range(3):
        r = pipe.process(make_signal(snr_db=15.0, t_ms=9000.0+i*100))
        _fmt(r)

    # ── Métriques finales ────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("▶ Métriques synaptiques finales\n")
    print(f"  Seuil LIF adaptatif          : {pipe.encoder.V_thresh:.2f} mV  "
          f"(cible : {BP.V_THRESH:.1f} mV)")
    print(f"  Taux de spike encoder        : {pipe.encoder._firing_rate_hz(pipe._t_ms):.2f} Hz  "
          f"(cible : {BP.R_TARGET_HZ:.1f} Hz)")
    print(f"  Compression (non-spike)      : {pipe.encoder.compression_ratio:.1%}")
    print(f"  Énergie totale encoder       : {pipe.encoder.total_energy:.4e} J")
    print(f"  LTP poids MUSIC (moy.)       : {np.mean(pipe.music.ltp_weights):.4f}")
    ltp_max_idx = np.argmax(pipe.music.ltp_weights)
    print(f"  Direction LTP renforcée      : {pipe.music.angle_grid[ltp_max_idx]:.1f}°")
    best_m = max(pipe.imm.MODELS, key=lambda m: pipe.imm.mu[pipe.imm.MODELS.index(m)])
    best_i = pipe.imm.MODELS.index(best_m)
    print(f"  Modèle IMM dominant (μ)      : {best_m}  "
          f"μ={pipe.imm.mu[best_i]:.4f}  w_syn={pipe.imm.w_syn[best_i]:.4f}")
    print(f"  Alertes spoofing totales     : {pipe.anomaly.n_alerts}")
    print(f"  Distribution ISI LTP count   : {pipe.anomaly.isi_ltp_count} trames")


if __name__ == "__main__":
    main() 