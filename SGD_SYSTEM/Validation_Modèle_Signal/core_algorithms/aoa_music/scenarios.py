"""
Ir divinkas — SYNAPTIC Lab
SGD System — Scénarios de Validation AOA

Définit les scénarios pour tester la précision de l'estimation de l'angle d'arrivée.
"""

import numpy as np
from signal_model import generate_array_signal

class AOAScenario:
    """Classe de base pour un scénario AOA."""
    def __init__(self, name, aoa_deg, amplitudes, snr_db):
        self.name = name
        self.aoa_deg = aoa_deg             # Liste des angles
        self.amplitudes = amplitudes       # Liste des amplitudes
        self.snr_db = snr_db
        self.n_sources = len(aoa_deg)

    def generate_received_signal(self, N, fs, d_lambda):
        """Génère le signal reçu sur le réseau d'antennes."""
        t = np.arange(N) / fs
        X = None
        
        # On simule N snapshots
        for i in range(self.n_sources):
            # Source : Bruit blanc complexe filtré ou sinusoïde pour test
            # Ici on utilise un signal aléatoire complexe pour le test de MUSIC
            s_i = (np.random.randn(N) + 1j * np.random.randn(N)) * self.amplitudes[i]
            
            # Propagation spatiale
            X_i = generate_array_signal(s_i, n_elements=2, d_lambda=d_lambda, aoa_deg=self.aoa_deg[i])
            
            if X is None:
                X = X_i
            else:
                X += X_i
        
        return X

class IdealAoaScenario(AOAScenario):
    """Scenario 1: Source unique, angle clair, SNR élevé."""
    def __init__(self):
        super().__init__("Source Unique (Idéal)", [20.0], [1.0], 20.0)

class LowSnrScenario(AOAScenario):
    """Scenario 2: Source unique, SNR limite (0 dB)."""
    def __init__(self):
        super().__init__("Bruit Élevé (0 dB)", [15.0], [1.0], 0.0)

class TwoSourcesScenario(AOAScenario):
    """Scenario 3: Deux sources séparées de 20°."""
    def __init__(self):
        super().__init__("Deux Sources (20° écart)", [-10.0, 10.0], [1.0, 1.0], 15.0)

class CloseSourcesScenario(AOAScenario):
    """Scenario 4: Deux sources proches (5° écart) - Défi pour MUSIC."""
    def __init__(self):
        super().__init__("Sources Proches (5° écart)", [0.0, 5.0], [1.0, 0.8], 20.0)

def get_aoa_scenarios():
    return [
        IdealAoaScenario(),
        LowSnrScenario(),
        TwoSourcesScenario(),
        CloseSourcesScenario()
    ]
