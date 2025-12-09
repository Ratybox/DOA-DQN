import numpy as np
from typing import Callable, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DOA:

    def __init__(
        self,
        dim: int,
        bounds: list,
        pop_size: int = 20,
        max_iter: int = 100,
        u: float = 0.9,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize DOA optimizer."""
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.u = u
        self.verbose = verbose
        
        # Verify bounds
        assert self.bounds.shape == (dim, 2), "bounds must have shape (dim, 2)"
        assert np.all(self.bounds[:, 0] < self.bounds[:, 1]), "bounds must satisfy lower < upper"
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
        
        # Exploration phase cutoff
        self.T_d = int(0.9 * max_iter)
        
        # Parameters for exploration phase (5 groups)
        self.k_q = self._compute_k_q()
        
        # Parameter for exploitation phase
        self.k_r = self._compute_k_r()
        
        # Tracking
        self.history_best = []
        self.population_history = []
        self.fitness_history = []
        
    def _compute_k_q(self) -> np.ndarray:
        """
        Compute number of forgetting dimensions for each group in exploration phase.
        Eq. (10) in the paper.
        """
        k_q = np.zeros(5, dtype=int)
        for q in range(5):
            q_id = q + 1  # groups are 1-indexed
            k_min = int(np.ceil(self.dim / (8.0 * q_id)))
            k_max = max(2, int(np.ceil(self.dim / (3.0 * q_id))))
            k_q[q] = self.rng.randint(k_min, k_max + 1)
        return k_q
    
    def _compute_k_r(self) -> int:
        """
        Compute number of forgetting dimensions for exploitation phase.
        Eq. (11) in the paper.
        """
        k_min = 2
        k_max = max(2, int(np.ceil(self.dim / 3.0)))
        return self.rng.randint(k_min, k_max + 1)
    
    def _initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize population uniformly in search space.
        Eq. (2)-(3) in the paper.
        """
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        population = low + self.rng.rand(self.pop_size, self.dim) * (high - low)
        return population
    
    def _clip_bounds(self, X: np.ndarray) -> np.ndarray:
        """Enforce bounds on population."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return np.clip(X, low, high)
    
    def _handle_boundary_violation(self, X: np.ndarray, i: int, j: int) -> float:
        """
        Handle boundary violations.
        Eq. (12) for Dim <= 15, Eq. (13) for Dim > 15.
        """
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        
        if self.dim <= 15:
            # Random reset method
            return low[j] + self.rng.rand() * (high[j] - low[j])
        else:
            # Dream-sharing inspired method
            m = self.rng.randint(0, self.pop_size)
            while m == i:
                m = self.rng.randint(0, self.pop_size)
            return X[m, j]
    
    def _exploration_phase(
        self,
        X: np.ndarray,
        fitness: np.ndarray,
        t: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exploration phase: divide population into 5 groups with different memory capabilities.
        """
        X_new = X.copy()
        pop_per_group = self.pop_size // 5
        
        for q in range(5):
            # Define group indices
            start_idx = q * pop_per_group
            if q == 4:  # Last group gets remainder
                end_idx = self.pop_size
            else:
                end_idx = (q + 1) * pop_per_group
            
            group_indices = np.arange(start_idx, end_idx)
            
            # Find best in group
            group_fitness = fitness[group_indices]
            best_group_idx = group_indices[np.argmin(group_fitness)]
            X_best_group = X[best_group_idx].copy()
            
            # Update each individual in the group
            for i in group_indices:
                # Memory strategy: Eq. (4)
                X_new[i] = X_best_group.copy()
                
                # Select forgetting dimensions
                k = self.k_q[q]
                forgetting_dims = self.rng.choice(self.dim, size=min(k, self.dim), replace=False)
                
                # Update forgotten dimensions
                for j in forgetting_dims:
                    if self.rng.rand() < self.u:
                        # Forgetting-supplementation strategy: Eq. (5)
                        low = self.bounds[j, 0]
                        high = self.bounds[j, 1]
                        
                        perturb = low + self.rng.rand() * (high - low)
                        alpha = 0.5 * (np.cos(np.pi * (t + self.max_iter - self.T_d) / self.max_iter) + 1.0)
                        
                        X_new[i, j] = X_best_group[j] + perturb * alpha
                    else:
                        # Dream-sharing strategy: Eq. (6)
                        m = self.rng.randint(0, self.pop_size)
                        if m <= i:
                            X_new[i, j] = X_new[m, j]
                        else:
                            X_new[i, j] = X[m, j]
        
        return X_new
    
    def _exploitation_phase(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Exploitation phase: entire population refines around global best.
        """
        X_new = np.tile(X_best, (self.pop_size, 1))
        
        # Select forgetting dimensions
        k = self.k_r
        forgetting_dims = self.rng.choice(self.dim, size=min(k, self.dim), replace=False)
        
        # Update forgotten dimensions for all individuals
        for i in range(self.pop_size):
            for j in forgetting_dims:
                low = self.bounds[j, 0]
                high = self.bounds[j, 1]
                
                perturb = low + self.rng.rand() * (high - low)
                alpha = 0.5 * (np.cos(np.pi * t / self.max_iter) + 1.0)
                
                X_new[i, j] = X_best[j] + perturb * alpha
        
        return X_new
    
    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Run the DOA optimization.
        
        Parameters
        ----------
        fitness_fn : Callable
            Objective function to minimize. Takes 1D array, returns scalar.
        callback : Optional[Callable]
            Callback function called after each iteration with (X, fitness, iteration).
        
        Returns
        -------
        best_X : np.ndarray
            Best solution found
        best_fitness : float
            Best fitness value
        info : dict
            Optimization information (history, etc.)
        """
        # Initialize population
        X = self._initialize_population()
        fitness = np.array([fitness_fn(x) for x in X])
        
        # Track best
        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        f_best = fitness[best_idx]
        
        # History
        self.history_best.append(f_best)
        self.population_history.append(X.copy())
        self.fitness_history.append(fitness.copy())
        
        if self.verbose:
            print(f"DOA Iteration 0//{self.max_iter} - Best Fitness: {f_best:.6e}")
        
        # Main optimization loop
        for t in range(self.max_iter):
            if t < self.T_d:
                # Exploration phase
                X_new = self._exploration_phase(X, fitness, t)
            else:
                # Exploitation phase
                X_new = self._exploitation_phase(X, X_best, t)
            
            # Handle bounds
            X_new = self._clip_bounds(X_new)
            
            # Evaluate new population
            fitness_new = np.array([fitness_fn(x) for x in X_new])
            
            # Elitism: keep better solutions (individual level)
            improved = fitness_new < fitness
            X[improved] = X_new[improved]
            fitness[improved] = fitness_new[improved]
            
            # Global elitism: update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < f_best:
                f_best = fitness[best_idx]
                X_best = X[best_idx].copy()
            
            # Track history
            self.history_best.append(f_best)
            self.population_history.append(X.copy())
            self.fitness_history.append(fitness.copy())
            
            if self.verbose and (t + 1) % max(1, self.max_iter // 10) == 0:
                print(f"DOA Iteration {t+1}//{self.max_iter} - Best Fitness: {f_best:.6e}")
            
            if callback is not None:
                callback(X, fitness, t + 1)
        
        info = {
            'history_best': np.array(self.history_best),
            'population_history': self.population_history,
            'fitness_history': self.fitness_history,
            'n_evals': self.pop_size * (self.max_iter + 1),
            'exploration_phase_end': self.T_d
        }
        
        return X_best, f_best, info


# ============================================================================
# Example usage and test functions
# ============================================================================

def sphere_function(x: np.ndarray) -> float:
    """Simple sphere function for testing."""
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function - multimodal."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


if __name__ == "__main__":
    # Test DOA on benchmark functions
    print("="*70)
    print("Testing Dream Optimization Algorithm (DOA)")
    print("="*70)
    
    # Test 1: Sphere function
    print("\n[TEST 1] Sphere Function (10D)")
    dim = 10
    bounds = [(-5.12, 5.12) for _ in range(dim)]
    
    doa = DOA(dim=dim, bounds=bounds, pop_size=20, max_iter=100, seed=42, verbose=True)
    best_x, best_f, info = doa.optimize(sphere_function)
    
    print(f"\nBest solution: {best_x}")
    print(f"Best fitness: {best_f:.6e}")
    print(f"Number of function evaluations: {info['n_evals']}")
    
    # Test 2: Rastrigin function
    print("\n" + "="*70)
    print("[TEST 2] Rastrigin Function (10D)")
    doa = DOA(dim=dim, bounds=bounds, pop_size=30, max_iter=200, seed=42, verbose=True)
    best_x, best_f, info = doa.optimize(rastrigin_function)
    
    print(f"\nBest solution (first 5): {best_x[:5]}...")
    print(f"Best fitness: {best_f:.6e}")
    print(f"Number of function evaluations: {info['n_evals']}")
    
    # Test 3: Rosenbrock function
    print("\n" + "="*70)
    print("[TEST 3] Rosenbrock Function (5D)")
    dim = 5
    bounds = [(-2.048, 2.048) for _ in range(dim)]
    
    doa = DOA(dim=dim, bounds=bounds, pop_size=25, max_iter=150, seed=42, verbose=True)
    best_x, best_f, info = doa.optimize(rosenbrock_function)
    
    print(f"\nBest solution: {best_x}")
    print(f"Best fitness: {best_f:.6e}")
    print(f"Number of function evaluations: {info['n_evals']}")
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(info['history_best'], 'b-', linewidth=2, label='DOA')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Fitness (log scale)', fontsize=12)
        ax.set_title('DOA Convergence Curve - Rosenbrock Function', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('doa_convergence.png', dpi=150)
        print("\n[PLOT] Convergence curve saved as 'doa_convergence.png'")
        plt.close()
    except ImportError:
        print("\n[INFO] Matplotlib not available for plotting")
