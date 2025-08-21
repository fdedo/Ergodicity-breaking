import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import time

def run_simulation(N, T, b, p, gamma_values, m, num_simulations, memory=True, output_dir="results"):
    """
    Run ensemble simulations of an imitation dynamics model with or without memory.

    Parameters:
    - N: number of agents
    - T: number of time steps
    - b: inverse temperature (sensitivity)
    - p: probability of external signal X being +1
    - gamma_values: array of gamma (social conformity weight)
    - m: memory window (used only if memory=True)
    - num_simulations: number of simulations per gamma
    - memory: use memory-based dynamics if True
    - output_dir: directory to save plot
    """

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results = []

    plt.figure(figsize=(10, 5))

    for gamma in gamma_values:
        temporal_means = []

        for sim in range(num_simulations):
            s = np.ones(N) if np.random.rand() < 0.5 else np.zeros(N)
            f_p_vec = []
            X_vec = []

            for t in range(T):
                f_p = np.mean(s)
                f_p_vec.append(f_p)

                X = 1 if np.random.rand() < p else -1
                X_vec.append(X)

                # Compute effective external signal
                if memory and t >= m:
                    X_eff = np.mean(X_vec[-m:])
                else:
                    X_eff = X

                # Payoffs
                p_plus = gamma * f_p + (1 - gamma) * X_eff
                p_minus = gamma * (1 - f_p) - (1 - gamma) * X_eff

                # Boltzmann probability to switch
                p_flip = np.exp(b * p_plus) / (np.exp(b * p_plus) + np.exp(b * p_minus))

                # Update agent states
                s = (np.random.rand(N) < p_flip).astype(int)

            # Compute temporal mean for this run
            temp_mean = np.mean(f_p_vec)
            temporal_means.append(temp_mean)

            # Blue: each simulation’s temporal average
            plt.plot(gamma, temp_mean, '+', color='b', markersize=8)

        overall_mean = np.mean(temporal_means)

        # Red: average across simulations
        plt.plot(gamma, overall_mean, '+', color='r', markersize=8)

        # Store results
        for tm in temporal_means:
            results.append((gamma, tm, overall_mean))

    # Plot formatting
    plt.xlabel(r'$\gamma$', fontsize=14)
    plt.ylabel(r'$\langle f_+ \rangle$', fontsize=14)
    plt.title(f'β = {b}, memory={"on" if memory else "off"}', fontsize=13)

    blue_dot = mlines.Line2D([], [], color='b', marker='+', linestyle='None', label='Temporal avg')
    red_dot = mlines.Line2D([], [], color='r', marker='+', linestyle='None', label='Ensemble avg')
    plt.legend(handles=[blue_dot, red_dot], loc='lower left', fontsize=12)

    # Save figure
    filename = f'{output_dir}/mean_beta{b}_{"memory" if memory else "memoryless"}_{N}agent_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return results

if __name__ == "__main__":
    # ---- Parameters ---- #
    N = 1
    T = 3000
    b = 100
    p = 0.6
    m = 1
    num_simulations = 1
    gamma_values = np.linspace(0, 1, 50)

    # Run with memory
    run_simulation(N, T, b, p, gamma_values, m, num_simulations, memory=True)
