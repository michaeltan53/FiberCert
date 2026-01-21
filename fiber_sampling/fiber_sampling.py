"""
Epsilon-Sampling Fiber (§3.3.3)

Definition 3.2: ε-sampling fiber
P̂_ε^N(b) = {y_i ~ π(·) : y_i ∈ Y_geo, ||G_c(y_i) - b|| ≤ ε}
"""

import numpy as np
from typing import Callable, Optional, List
from ..models.hierarchical_model import ScenarioMapping


class EpsilonSamplingFiber:
    """
    ε-sampling fiber implementation
    
    P̂_ε^N(b) = {y_i ~ π(·) : y_i ∈ Y_geo, ||G_c(y_i) - b|| ≤ ε}
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        proposal_distribution: Callable[[int], np.ndarray],
        epsilon: float = 0.1,
        pi_min: float = 0.01,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            proposal_distribution: Function π(·) that samples n points
            epsilon: Geometric tolerance ε
            pi_min: Lower bound π_min on proposal mass in fiber neighborhood
        """
        self.scenario_mapping = scenario_mapping
        self.proposal_distribution = proposal_distribution
        self.epsilon = epsilon
        self.pi_min = pi_min
    
    def sample(
        self,
        b: np.ndarray,
        n: int,
        max_attempts: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample ε-sampling fiber P̂_ε^N(b)
        
        Args:
            b: Observed behavior
            n: Number of samples N
            max_attempts: Maximum sampling attempts (None = no limit)
            
        Returns:
            Array of parameter points in fiber
        """
        if max_attempts is None:
            max_attempts = n * 10  # Default: try 10x more samples
        
        fiber_samples = []
        attempts = 0
        
        while len(fiber_samples) < n and attempts < max_attempts:
            # Sample from proposal distribution
            y_candidates = self.proposal_distribution(n - len(fiber_samples))
            
            for y in y_candidates:
                # Check if in fiber neighborhood
                b_pred = self.scenario_mapping(y)
                if np.linalg.norm(b_pred - b) <= self.epsilon:
                    fiber_samples.append(y)
                    
                    if len(fiber_samples) >= n:
                        break
            
            attempts += len(y_candidates)
        
        if len(fiber_samples) < n:
            # Warning: not enough samples found
            pass
        
        return np.array(fiber_samples) if fiber_samples else np.array([]).reshape(0, len(b))
    
    def hit_rate(self, b: np.ndarray, n_test: int = 1000) -> float:
        """
        Estimate hit rate π(V_ε(b)) for monitoring
        
        Args:
            b: Behavior point
            n_test: Number of test samples
            
        Returns:
            Estimated hit rate
        """
        y_samples = self.proposal_distribution(n_test)
        hits = 0
        
        for y in y_samples:
            b_pred = self.scenario_mapping(y)
            if np.linalg.norm(b_pred - b) <= self.epsilon:
                hits += 1
        
        return hits / n_test


def sample_fiber(
    b: np.ndarray,
    scenario_mapping: ScenarioMapping,
    proposal_distribution: Callable[[int], np.ndarray],
    n: int,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Convenience function for fiber sampling
    
    Args:
        b: Observed behavior
        scenario_mapping: Mapping G_c
        proposal_distribution: Proposal distribution π(·)
        n: Number of samples
        epsilon: Geometric tolerance
        
    Returns:
        Array of parameter points in fiber
    """
    fiber_sampler = EpsilonSamplingFiber(
        scenario_mapping=scenario_mapping,
        proposal_distribution=proposal_distribution,
        epsilon=epsilon,
    )
    return fiber_sampler.sample(b, n)

