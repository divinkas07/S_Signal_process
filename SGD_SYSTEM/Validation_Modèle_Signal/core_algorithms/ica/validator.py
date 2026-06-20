"""
Ir divinkas — SYNAPTIC Lab
SGD System — Validateur ICA (Monte Carlo)

Gère l'exécution des simulations Monte Carlo pour la séparation de sources.
"""

import numpy as np
from tqdm import tqdm
from signal_model import generate_array_signal
from channel_model import generate_awgn
from metrics import compute_sir
from .estimator import estimate_ica

class ICAValidator:
    """
    Exécute les simulations de validation pour l'ICA.
    """
    def __init__(self, config):
        self.config = config
        self.fs = config['signal']['fs']
        self.arr = config['array']

    def run_scenario(self, scenario, n_runs=100, snr_db=20):
        """
        Lance une série de runs Monte Carlo pour un scénario donné.
        """
        N = int(self.fs * 0.01) # 10ms par défaut pour tests
        sir_results = []
        success_count = 0
        
        print(f"\n>> Scenario : {scenario.name} ({n_runs} runs, SNR={snr_db}dB)")
        
        for _ in tqdm(range(n_runs), desc="Simulation"):
            # 1. Génération des sources
            S_true = scenario.generate_sources(N, self.fs)
            n_sources = S_true.shape[0]
            
            # 2. Mélange spatial (ULA)
            X = np.zeros((self.arr['n_elements'], N), dtype=complex)
            for k in range(n_sources):
                s_k = S_true[k] * scenario.amplitudes[k]
                X += generate_array_signal(s_k, self.arr['n_elements'], 
                                          self.arr['d_lambda'], scenario.aoa_deg[k])
            
            # 3. Ajout du bruit
            p_sig = np.mean(np.abs(X)**2)
            noise = generate_awgn(X.shape, snr_db, p_sig)
            
            # Cas spécifique : Bruit Impulsionnel
            if scenario.name == "Bruit Impulsionnel":
                n_impulses = int(0.01 * X.size)
                idx = np.random.choice(X.size, n_impulses, replace=False)
                noise.ravel()[idx] += (np.random.randn(n_impulses) + 1j*np.random.randn(n_impulses)) * np.sqrt(p_sig) * 10
            
            X_noisy = X + noise
            
            # 4. Séparation ICA
            try:
                S_est, _ = estimate_ica(X_noisy, n_components=n_sources)
                
                # 5. Calcul des métriques (SIR)
                sirs, _ = compute_sir(S_true, S_est)
                mean_sir = np.mean(sirs)
                sir_results.append(mean_sir)
                
                if mean_sir > 10.0:
                    success_count += 1
            except Exception as e:
                sir_results.append(0)
                
        avg_sir = np.mean(sir_results)
        success_rate = (success_count / n_runs) * 100
        
        return {
            "avg_sir": avg_sir,
            "success_rate": success_rate,
            "results": sir_results
        }
