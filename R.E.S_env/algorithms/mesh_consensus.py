"""
Mesh Consensus – Distributed consensus algorithm for multi-node fusion.
Multiple receiving nodes share their local estimates and converge
to a common fused estimate via average consensus iterations.
"""

import numpy as np


class MeshConsensus:
    """
    Average consensus protocol for fusing estimates
    from multiple receiving nodes in a mesh network.
    
    Each node starts with a local estimate, and through iterative
    averaging with neighbors, all nodes converge to the global average.
    """

    def __init__(self, n_nodes: int = 4, n_iterations: int = 10,
                 connectivity: float = 0.7):
        self.n_nodes = n_nodes
        self.n_iterations = n_iterations
        self.connectivity = connectivity
        self._topology = self._generate_topology()
        self._metrics = {}

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: fuse estimates from multiple nodes.
        
        Expects:
            'aoa_estimate_deg': primary node AOA estimate
            
        Adds:
            'consensus_estimate': fused estimate after consensus
            'consensus_convergence': convergence history
        """
        primary_aoa = data.get("aoa_estimate_deg", [0.0])
        if isinstance(primary_aoa, list):
            primary_aoa = primary_aoa[0] if primary_aoa else 0.0

        # Simulate local estimates at each node with noise
        local_estimates = self._simulate_local_estimates(
            primary_aoa, noise_std=2.0
        )

        # Run consensus
        fused, history = self.run_consensus(local_estimates)

        data["consensus_estimate"] = float(fused)
        data["consensus_node_estimates"] = local_estimates.tolist()
        data["consensus_history"] = history

        self._metrics = {
            "fused_aoa": float(fused),
            "local_spread": float(np.std(local_estimates)),
            "converged_spread": float(np.std(history[-1]) if history else 0),
            "n_iterations": self.n_iterations,
            "status": "success",
        }
        return data

    def run_consensus(self, local_estimates: np.ndarray) -> tuple:
        """
        Run average consensus protocol.
        
        Each iteration: xᵢ(k+1) = Σⱼ wᵢⱼ · xⱼ(k)
        where wᵢⱼ = 1/degree(i) if connected, 0 otherwise.
        
        Returns:
            (fused_estimate, convergence_history)
        """
        x = local_estimates.copy()
        history = [x.copy()]

        # Metropolis weights
        W = self._compute_weights()

        for _ in range(self.n_iterations):
            x_new = W @ x
            x = x_new
            history.append(x.copy())

        # Final fused estimate is the average
        fused = np.mean(x)
        return fused, history

    def _compute_weights(self) -> np.ndarray:
        """
        Compute Metropolis-Hastings doubly stochastic weight matrix.
        Ensures convergence to the global average.
        """
        A = self._topology  # Adjacency matrix
        n = self.n_nodes
        W = np.zeros((n, n))

        for i in range(n):
            neighbors = np.where(A[i, :] > 0)[0]
            di = len(neighbors)
            for j in neighbors:
                dj = np.sum(A[j, :])
                W[i, j] = 1.0 / (max(di, dj) + 1)
            W[i, i] = 1.0 - np.sum(W[i, :])

        return W

    def _generate_topology(self) -> np.ndarray:
        """Generate random mesh network topology."""
        A = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if np.random.rand() < self.connectivity:
                    A[i, j] = 1
                    A[j, i] = 1

        # Ensure connected: connect isolated nodes to node 0
        for i in range(self.n_nodes):
            if np.sum(A[i, :]) == 0 and i != 0:
                A[i, 0] = 1
                A[0, i] = 1

        return A

    def _simulate_local_estimates(self, true_value: float,
                                   noise_std: float = 2.0) -> np.ndarray:
        """Simulate noisy local estimates at each node."""
        return true_value + np.random.randn(self.n_nodes) * noise_std

    def set_topology(self, adjacency: np.ndarray):
        """Set custom network topology (adjacency matrix)."""
        self._topology = adjacency
        self.n_nodes = adjacency.shape[0]

    def get_metrics(self) -> dict:
        return self._metrics
