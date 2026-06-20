"""
Ir divinkas - SYNAPTIC Lab
SGD System - Validation de la Separation de Sources (ICA)

Script de validation Monte Carlo couvrant 4 scenarios critiques :
1. Cas Ideal (2 sources, SNR eleve)
2. Brouillage Fort (Source + Jammer +3dB)
3. Sources Correlees (Degradation attendue)
4. Bruit Impulsionnel (Robustesse)

Metriques : SIR Gain, Taux de succes, Stabilite
"""

import sys
from pathlib import Path

# Adjust sys.path to allow absolute imports from the root if run directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator_engine.signal_model import load_config, generate_time_vector, generate_array_signal
from simulator_engine.channel_model import apply_channel, generate_awgn, signal_power
from core_algorithms.estimator_ica import estimate_ica
from core_algorithms.metrics import compute_sir

def generate_scenario_signals(scenario_id, N, fs):
    """
    Genere les sources (S) et la configuration de melange pour un scenario donne.
    Returns: S_true, aoa_deg, amplitudes
    """
    t = np.arange(N) / fs
    
    if scenario_id == 1: # Cas Ideal
        # 2 sources independantes, puissances egales
        # Source 1: QPSK
        s1 = (np.random.randint(0, 2, N)*2 - 1) + 1j*(np.random.randint(0, 2, N)*2 - 1)
        s1 = s1 / np.std(s1)
        # Source 2: Multi-tone
        s2 = np.exp(1j * 2 * np.pi * 1e3 * t)
        
        S_true = np.vstack([s1, s2])
        aoa_deg = [10, 40]
        amplitudes = [1.0, 1.0]

    elif scenario_id == 2: # Brouillage Fort
        # Source 1: Signal d'intérêt (QPSK)
        s1 = (np.random.randint(0, 2, N)*2 - 1) + 1j*(np.random.randint(0, 2, N)*2 - 1)
        # Source 2: Jammer (Bruit coloré ou FM large bande)
        # Modélisons un jammer par un bruit filtré (centré ailleurs) ou juste un bruit fort
        s2 = (np.random.randn(N) + 1j*np.random.randn(N))
        
        # Normalisation
        s1 = s1 / np.std(s1)
        s2 = s2 / np.std(s2)
        
        S_true = np.vstack([s1, s2])
        aoa_deg = [10, 20] # Rapprochés pour compliquer
        # Jammer à +3dB -> amplitude factor sqrt(10^(3/10)) approx 1.41
        amplitudes = [1.0, 1.41] 
        
    elif scenario_id == 3: # Sources Correlees
        # Source 1
        s1 = np.random.randn(N) + 1j*np.random.randn(N)
        # Source 2 = alpha * s1 + (1-alpha) * noise
        corr_factor = 0.6
        noise = np.random.randn(N) + 1j*np.random.randn(N)
        s2 = corr_factor * s1 + np.sqrt(1 - corr_factor**2) * noise
        
        S_true = np.vstack([s1, s2])
        aoa_deg = [-15, 15]
        amplitudes = [1.0, 1.0]
        
    elif scenario_id == 4: # Bruit Impulsionnel
        # Comme scenario 1 mais on gerera le bruit apres
        s1 = np.exp(1j * 2 * np.pi * 500 * t)
        s2 = np.exp(1j * 2 * np.pi * 1500 * t)
        
        S_true = np.vstack([s1, s2])
        aoa_deg = [30, 60]
        amplitudes = [1.0, 1.0]
        
    else:
        raise ValueError("Unknown scenario")
        
    return S_true, aoa_deg, amplitudes

def run_simulation(scenario_id, n_runs=100, snr_db=20):
    cfg = load_config()
    fs = cfg['signal']['fs']
    duration = 0.01 # 10ms pour tests rapides
    N = int(fs * duration)
    
    arr = cfg['array']
    n_elements = arr['n_elements']
    d_lambda = arr['d_lambda']
    
    print(f"\n[Scenario {scenario_id}] ({n_runs} runs) - SNR={snr_db}dB")
    
    sir_gains = []
    success_count = 0
    
    for i in tqdm(range(n_runs)):
        # 1. Génération Sources
        S_true, aoa_deg, amps = generate_scenario_signals(scenario_id, N, fs)
        n_sources = S_true.shape[0]
        
        # 2. Melange (Simule via channel_model ou manuellement pour multiples AOAs)
        # channel_model.py gere UNE aoa pour UN signal. Ici on melange manuellement.
        X = np.zeros((n_elements, N), dtype=complex)
        
        for k in range(n_sources):
            # Generation vecteur directionnel
            # On utilise generate_array_signal pour chaque source individuellement
            # Note: generate_array_signal attend un signal 1D
            s_k = S_true[k] * amps[k]
            x_k = generate_array_signal(s_k, n_elements, d_lambda, aoa_deg[k])
            X += x_k
            
        # 3. Ajout Bruit / Canal
        # Calcul puissance signal total moyen par antenne
        p_sig = np.mean(np.abs(X)**2)
        
        if scenario_id == 4:
            # Bruit Impulsionnel : Poisson process ou outliers
            # On ajoute un bruit gaussien de fond + des pics
            noise = generate_awgn(X.shape, snr_db, p_sig)
            
            # Impulsions : 1% des samples affectés par un bruit x100
            n_impulses = int(0.01 * X.size)
            idx_flat = np.random.choice(X.size, n_impulses, replace=False)
            noise.ravel()[idx_flat] += (np.random.randn(n_impulses) + 1j*np.random.randn(n_impulses)) * np.sqrt(p_sig) * 10
            
            X_noisy = X + noise
        else:
            # Bruit AWGN standard
            noise = generate_awgn(X.shape, snr_db, p_sig)
            X_noisy = X + noise
            
        # Calcul SIR Input (approx, car mélange)
        # Difficile à définir "SIR in" globalement sans connaitre le beamforming.
        # On se base sur le SIR Output vs 0dB (Amélioration relative)
        # Ou on compare SIR_out à SIR_in (si on considère une antenne omni ?)
        # Pour simplifier KPI : On regarde le SIR Output absolu.
            
        # 4. Séparation ICA
        try:
            S_est, W, _ = estimate_ica(X_noisy, n_components=n_sources, max_iter=1000)
            
            # 5. Métriques
            sirs_out, _ = compute_sir(S_true, S_est)
            
            # On moyenne les SIR des sources (ou min ?)
            mean_sir = np.mean(sirs_out)
            sir_gains.append(mean_sir)
            
            # Succès si SIR moyen > 10dB (pour les cas nominaux)
            if mean_sir > 10.0:
                success_count += 1
                
        except Exception as e:
            print(f"Run {i} failed: {e}")
            sir_gains.append(0)

    # Stats
    avg_sir = np.mean(sir_gains)
    success_rate = success_count / n_runs * 100
    
    print(f"  * Resultats Scenario {scenario_id}:")
    print(f"     SIR Moyen Output : {avg_sir:.2f} dB")
    print(f"     Taux Succes (>10dB) : {success_rate:.1f}%")
    
    return avg_sir, success_rate

if __name__ == "__main__":
    # Paramètres globaux
    N_RUNS = 50 
    
    # Rapport
    report = {}
    
    # Scénario 1 : Idéal (SNR 20dB)
    s1_sir, s1_rate = run_simulation(1, N_RUNS, snr_db=20)
    report['Scenario 1 (Ideal)'] = {'SIR': s1_sir, 'Success': s1_rate}
    
    # Scénario 2 : Jammer (SNR 5dB pour le signal utile, Jammer fort)
    # Note: SNR ici est le ratio SignalTotal / Bruit thermique.
    # Le SIR (Signal/Jammer) est géré par les amplitudes.
    s2_sir, s2_rate = run_simulation(2, N_RUNS, snr_db=15)
    report['Scenario 2 (Jammer)'] = {'SIR': s2_sir, 'Success': s2_rate}
    
    # Scénario 3 : Corrélé (SNR 20dB)
    s3_sir, s3_rate = run_simulation(3, N_RUNS, snr_db=20)
    report['Scenario 3 (Correlated)'] = {'SIR': s3_sir, 'Success': s3_rate}
    
    # Scénario 4 : Impulsionnel
    s4_sir, s4_rate = run_simulation(4, N_RUNS, snr_db=20)
    report['Scenario 4 (Impulsive)'] = {'SIR': s4_sir, 'Success': s4_rate}
    
    print("\n" + "="*40)
    print("RESUME GLOBAL VALIDATION ICA")
    print("="*40)
    for k, v in report.items():
        print(f"{k:25s} | SIR: {v['SIR']:5.1f} dB | OK: {v['Success']:5.1f}%")
        
    # Décision GO/NO-GO
    if report['Scenario 1 (Ideal)']['Success'] > 95:
        print("\n✅ GO pour intégration hardware (sur base simu)")
    else:
        print("\n❌ NO-GO : Optimisation requise")
