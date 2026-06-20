"""
Ir divinkas — SYNAPTIC Lab
SGD System — Validateur AOA (Monte Carlo)

Gère l'exécution des simulations Monte Carlo pour la validation de l'AOA.
"""

import numpy as np
from tqdm import tqdm
from channel_model import generate_awgn
from crlb import crlb_aoa
from .estimator import estimate_aoa

class AOAValidator:
    """
    Exécute les simulations de validation pour l'estimation AOA.
    """
    def __init__(self, config):
        self.config = config
        self.fs = config['signal']['fs']
        self.arr = config['array']

    def run_scenario(self, scenario, n_runs=100):
        """
        Exécute n_runs Monte Carlo pour un scénario donné.
        """
        N = int(self.fs * 0.01) # 10ms d'observation
        errors = []
        success_count = 0
        
        print(f"\n>> Validant AOA : {scenario.name}")
        
        for _ in tqdm(range(n_runs), desc="Monte Carlo", leave=False):
            # 1. Génération du mélange
            X = scenario.generate_received_signal(N, self.fs, self.arr['d_lambda'])
            
            # 2. Ajout du bruit
            p_sig = np.mean(np.abs(X)**2)
            noise = generate_awgn(X.shape, scenario.snr_db, p_sig)
            X_noisy = X + noise
            
            # 3. Estimation
            try:
                aoa_est = estimate_aoa(X_noisy, scenario.n_sources, self.arr['d_lambda'])
                
                # 4. Calcul de l'erreur (appariement simple pour 1 ou 2 sources)
                aoa_true = np.sort(scenario.aoa_deg)
                
                if np.any(np.isnan(aoa_est)):
                    errors.append(np.nan)
                    continue
                
                # Erreur absolue moyenne sur les sources
                err = np.abs(aoa_est - aoa_true)
                errors.append(np.mean(err))
                
                # Succès si l'erreur moyenne < 2.0°
                if np.mean(err) < 2.0:
                    success_count += 1
                    
            except Exception as e:
                errors.append(np.nan)
        
        errors = np.array(errors)
        valid_errors = errors[~np.isnan(errors)]
        
        rmse = np.sqrt(np.mean(valid_errors**2)) if len(valid_errors) > 0 else np.inf
        bias = np.mean(valid_errors) if len(valid_errors) > 0 else np.inf
        variance = np.var(valid_errors) if len(valid_errors) > 0 else np.inf
        success_rate = (success_count / n_runs) * 100
        
        # Calcul CRLB théorique pour comparaison (moyenne si multi-source)
        theoretical_crlb = np.mean([
            crlb_aoa(self.arr['n_elements'], self.arr['d_lambda'], scenario.snr_db, a, n_snapshots=N)
            for a in scenario.aoa_deg
        ])
        
        return {
            "name": scenario.name,
            "rmse": rmse,
            "bias": bias,
            "variance": variance,
            "crlb": theoretical_crlb,
            "success_rate": success_rate,
            "ratio_crlb": rmse / theoretical_crlb if theoretical_crlb > 0 else np.inf
        }
