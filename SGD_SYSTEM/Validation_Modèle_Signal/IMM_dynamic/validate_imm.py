import numpy as np
import matplotlib.pyplot as plt
from imm_filter import IMMFilter
import time

def generate_trajectory(steps=200, dt=0.1, noise_std=0.5):
    """
    Generate a trajectory with 4 phases: Static, Walking, Running, Erratic.
    Returns:
        true_states: Array of [x, y, vx, vy]
        measurements: Array of [x, y]
        mode_labels: Array of mode indices (0: Static, 1: Walk, 2: Run, 3: Erratic)
    """
    true_states = []
    mode_labels = []
    x = np.array([0.0, 0.0, 0.0, 0.0])
    
    for k in range(steps):
        if k < 50:        # Static
            mode = 0
            v = 0.0
            vx, vy = 0.0, 0.0
        elif k < 100:     # Walking
            mode = 1
            v = 1.4
            vx, vy = v, 0.0
        elif k < 150:     # Running
            mode = 2
            v = 3.0
            vx, vy = v, 0.0
        else:             # Erratic
            mode = 3
            vx = np.random.randn() * 2.0
            vy = np.random.randn() * 2.0
        
        # Update state (simple CV model for simulation)
        x[0] += vx * dt
        x[1] += vy * dt
        x[2] = vx
        x[3] = vy
        
        true_states.append(x.copy())
        mode_labels.append(mode)
        
    true_states = np.array(true_states)
    measurements = true_states[:, :2] + np.random.randn(steps, 2) * noise_std
    
    return true_states, measurements, np.array(mode_labels)

def calculate_kpis(prob_history, mode_labels, dt=0.1):
    """
    Calculate KPIs based on probability history.
    """
    steps = len(mode_labels)
    correct_prob_sum = 0
    oscillations = 0
    convergence_times = []
    
    last_dominant = -1
    
    # Identify phase transitions
    phase_starts = [0]
    for i in range(1, steps):
        if mode_labels[i] != mode_labels[i-1]:
            phase_starts.append(i)
    phase_starts.append(steps)
    
    for p in range(len(phase_starts)-1):
        start, end = phase_starts[p], phase_starts[p+1]
        target_mode = mode_labels[start]
        
        # Convergence time
        converged_idx = -1
        for i in range(start, end):
            if prob_history[i, target_mode] > 0.8:
                converged_idx = i
                break
        
        if converged_idx != -1:
            convergence_times.append((converged_idx - start) * dt)
        else:
            convergence_times.append((end - start) * dt) # Failed to converge in phase
            
        # Dominant model and oscillations within phase
        phase_dominant_changes = 0
        current_dominant = -1
        
        for i in range(start, end):
            dominant = np.argmax(prob_history[i])
            if dominant == target_mode:
                correct_prob_sum += 1
            
            if dominant != current_dominant:
                if current_dominant != -1: # Ignore first setting of dominant in phase
                    phase_dominant_changes += 1
                current_dominant = dominant
        
        # We subtract transitions between phases from total oscillations if we wanted total stability
        # But here we count changes WITHIN phases as instability
        oscillations += phase_dominant_changes

    avg_convergence = np.mean(convergence_times) if convergence_times else 0
    prob_correct = (correct_prob_sum / steps) * 100
    
    return avg_convergence, prob_correct, oscillations

def run_monte_carlo(n_runs=1000, steps=200, dt=0.1):
    print(f"Starting Monte Carlo simulation with {n_runs} runs...")
    
    total_conv = 0
    total_prob = 0
    total_osc = 0
    divergence_count = 0
    
    start_time = time.time()
    
    for run in range(n_runs):
        if (run + 1) % 100 == 0:
            print(f"Run {run + 1}/{n_runs}...")
            
        true_states, measurements, mode_labels = generate_trajectory(steps, dt)
        imm_filter = IMMFilter(dt)
        prob_history = []
        
        try:
            for z in measurements:
                imm_filter.predict()
                imm_filter.update(z)
                prob_history.append(imm_filter.mu)
                
                # Simple check for numerical divergence
                if np.any(np.isnan(imm_filter.x)):
                    raise ValueError("Numerical divergence")
            
            prob_history = np.array(prob_history)
            conv, prob, osc = calculate_kpis(prob_history, mode_labels, dt)
            
            total_conv += conv
            total_prob += prob
            total_osc += osc
            
        except Exception as e:
            divergence_count += 1
            continue
            
    end_time = time.time()
    
    valid_runs = n_runs - divergence_count
    if valid_runs == 0:
        print("All runs diverged!")
        return
        
    avg_conv = total_conv / valid_runs
    avg_prob = total_prob / valid_runs
    avg_osc = total_osc / valid_runs
    
    print("\n" + "="*30)
    print("MONTE CARLO RESULTS")
    print("="*30)
    print(f"Total Runs: {n_runs}")
    print(f"Valid Runs: {valid_runs}")
    print(f"Divergences: {divergence_count}")
    print(f"Avg Convergence Time: {avg_conv:.3f} s")
    print(f"Avg Correct Mode Prob: {avg_prob:.1f} %")
    print(f"Avg Oscillations/Run: {avg_osc:.1f}")
    print(f"Processing Time: {end_time - start_time:.2f} s")
    print("="*30)
    
    # Validation criteria check
    print("\nCRITERIA CHECK:")
    print(f"- Convergence < 0.5s: {'OK' if avg_conv < 0.5 else 'FAIL'}")
    print(f"- Prob Correct > 90%: {'OK' if avg_prob > 90 else 'FAIL'}")
    print(f"- Oscillations < 5:   {'OK' if avg_osc < 5 else 'FAIL'}")
    print(f"- Divergence = 0:     {'OK' if divergence_count == 0 else 'FAIL'}")
    
    if avg_conv < 0.5 and avg_prob > 90 and avg_osc < 5 and divergence_count == 0:
        print("\n>>> STATUS: GO TO HARDWARE INTEGRATION! <<<")
    else:
        print("\n>>> STATUS: NO-GO. Tuning required. <<<")

def single_run_visualization(steps=200, dt=0.1):
    true_states, measurements, mode_labels = generate_trajectory(steps, dt)
    imm_filter = IMMFilter(dt)
    
    prob_history = []
    estimates = []
    
    for z in measurements:
        imm_filter.predict()
        imm_filter.update(z)
        prob_history.append(imm_filter.mu)
        estimates.append(imm_filter.x)
        
    prob_history = np.array(prob_history)
    estimates = np.array(estimates)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Trajectory plot
    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Trajectory', alpha=0.5)
    plt.scatter(measurements[:, 0], measurements[:, 1], c='r', s=2, label='Measurements', alpha=0.3)
    plt.plot(estimates[:, 0], estimates[:, 1], 'b--', label='IMM Estimate')
    plt.title("Trajectory Tracking")
    plt.legend()
    plt.grid(True)
    
    # Probability plot
    plt.subplot(2, 1, 2)
    plt.plot(prob_history[:, 0], label="Static")
    plt.plot(prob_history[:, 1], label="Walk")
    plt.plot(prob_history[:, 2], label="Run")
    plt.plot(prob_history[:, 3], label="Erratic")
    
    # Highlight true modes
    for k in range(steps):
        mode = mode_labels[k]
        plt.axvspan(k, k+1, color=plt.cm.tab10(mode), alpha=0.1)
        
    plt.xlabel("Steps")
    plt.ylabel("Probability")
    plt.title("IMM Model Probabilities")
    plt.legend()
    plt.ylim([0, 1.1])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("imm_validation_single_run.png")
    print("\nSaved single run visualization to imm_validation_single_run.png")
    # plt.show() # Can't show window in this environment

if __name__ == "__main__":
    # Perform one visualization run
    single_run_visualization()
    
    # Perform Monte Carlo validation
    run_monte_carlo(n_runs=1000)
