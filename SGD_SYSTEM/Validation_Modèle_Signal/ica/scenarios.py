"""
Ir divinkas — SYNAPTIC Lab
SGD System — Scénarios de Validation ICA

Définit les différents scénarios de test pour évaluer la robustesse de l'ICA.
"""

import numpy as np
from signal_model import generate_array_signal

class Scenario:
    """Classe de base pour un scénario de test."""
    def __init__(self, name, n_sources, aoa_deg, amplitudes):
        self.name = name
        self.n_sources = n_sources
        self.aoa_deg = aoa_deg
        self.amplitudes = amplitudes

    def generate_sources(self, N, fs):
        """Doit retourner une matrice S (n_sources, N)."""
        raise NotImplementedError

class IdealScenario(Scenario):
    """Scenario 1: 2 sources indépendantes, SNR élevé."""
    def __init__(self):
        super().__init__("Cas Idéal", 2, [10, 40], [1.0, 1.0])

    def generate_sources(self, N, fs):
        t = np.arange(N) / fs
        # Source 1: QPSK symbol-like noise
        s1 = (np.random.randint(0, 2, N)*2 - 1) + 1j*(np.random.randint(0, 2, N)*2 - 1)
        # Source 2: Sinusoïde pure
        s2 = np.exp(1j * 2 * np.pi * 1000 * t)
        return np.vstack([s1/np.std(s1), s2/np.std(s2)])

class JammerScenario(Scenario):
    """Scenario 2: Signal utile + Brouilleur fort (+3dB)."""
    def __init__(self):
        super().__init__("Brouillage Fort", 2, [10, 20], [1.0, 1.41])

    def generate_sources(self, N, fs):
        # Source 1: Signal utile
        s1 = (np.random.randint(0, 2, N)*2 - 1) + 1j*(np.random.randint(0, 2, N)*2 - 1)
        # Source 2: Brouilleur (Bruit blanc complexe)
        s2 = np.random.randn(N) + 1j*np.random.randn(N)
        return np.vstack([s1/np.std(s1), s2/np.std(s2)])

class CorrelatedScenario(Scenario):
    """Scenario 3: Sources partiellement corrélées."""
    def __init__(self, correlation=0.6):
        super().__init__("Sources Corrélées", 2, [-15, 15], [1.0, 1.0])
        self.correlation = correlation

    def generate_sources(self, N, fs):
        s1 = np.random.randn(N) + 1j*np.random.randn(N)
        noise = np.random.randn(N) + 1j*np.random.randn(N)
        s2 = self.correlation * s1 + np.sqrt(1 - self.correlation**2) * noise
        return np.vstack([s1/np.std(s1), s2/np.std(s2)])

class ImpulsiveScenario(Scenario):
    """Scenario 4: Présence de pics de bruit impulsionnels."""
    def __init__(self):
        super().__init__("Bruit Impulsionnel", 2, [30, 60], [1.0, 1.0])

    def generate_sources(self, N, fs):
        t = np.arange(N) / fs
        s1 = np.exp(1j * 2 * np.pi * 500 * t)
        s2 = np.exp(1j * 2 * np.pi * 1500 * t)
        return np.vstack([s1/np.std(s1), s2/np.std(s2)])

def get_all_scenarios():
    """Retourne la liste des scénarios à tester."""
    return [
        IdealScenario(),
        JammerScenario(),
        CorrelatedScenario(),
        ImpulsiveScenario()
    ]
