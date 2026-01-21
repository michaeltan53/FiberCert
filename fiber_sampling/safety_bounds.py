"""
Safety-Side Upper Bounds (§3.3.3)

Theorem 3.1: Safety-side upper bound via fiber sampling
U_prim(b) = max_{y ∈ P̂_ε^N(b)} LLR_Y(y) + Δ_geom(ε) + Δ_sample + Δ_tail
"""

import numpy as np
from typing import Optional, Callable
from ..models.hierarchical_model import ScenarioMapping
from ..models.generative_models import IdentityModel, BackgroundModel
from .fiber_sampling import EpsilonSamplingFiber


class SafetyBound:
    """
    Safety-side upper bound computation
    
    Implements Theorem 3.1
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        identity_model: IdentityModel,
        background_model: BackgroundModel,
        identity: str,
        context: Optional[str] = None,
        epsilon: float = 0.1,
        L_LLR: float = 2.5,
        delta_geom: float = 0.05,
        delta_sample: float = 0.01,
        sigma_eff: float = 0.02,
        C_cov: Optional[float] = None,
        d_joint: Optional[float] = None,
        pi_min: float = 0.01,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            identity_model: P_θ
            background_model: P_u
            identity: Identity z
            context: Context c
            epsilon: Geometric tolerance ε
            L_LLR: Lipschitz constant for LLR_Y
            delta_geom: Geometric failure probability δ_geom
            delta_sample: Sampling failure probability δ_sample
            sigma_eff: Effective tail parameter σ_c^eff
            C_cov: Covering constant C_cov(d_joint)
            d_joint: Joint dimension d_joint
            pi_min: Lower bound on proposal mass
        """
        self.scenario_mapping = scenario_mapping
        self.identity_model = identity_model
        self.background_model = background_model
        self.identity = identity
        self.context = context
        self.epsilon = epsilon
        self.L_LLR = L_LLR
        self.delta_geom = delta_geom
        self.delta_sample = delta_sample
        self.sigma_eff = sigma_eff
        self.C_cov = C_cov if C_cov is not None else 1.0
        self.d_joint = d_joint if d_joint is not None else 2.0
        self.pi_min = pi_min
    
    def compute_llr_y(self, y: np.ndarray) -> float:
        """
        Compute parameter space LLR
        
        LLR_Y(y) = log(P_θ(y|z,c) / P_u(y|c))
        """
        log_p_theta = self.identity_model.log_prob(y, z=self.identity, c=self.context)
        log_p_u = self.background_model.log_prob(y, z=None, c=self.context)
        
        if log_p_u == -np.inf:
            return np.inf
        
        return log_p_theta - log_p_u
    
    def compute_geometric_error(self) -> float:
        """
        Compute geometric approximation error
        
        Δ_geom(ε) ≤ L_LLR · ε
        """
        return self.L_LLR * self.epsilon
    
    def compute_sample_error(self, n: int) -> float:
        """
        Compute sampling error
        
        Δ_sample ≤ L_LLR · (C_cov · log(2N/δ_sample) / (N · π_min))^(1/d_joint)
        """
        if n == 0:
            return np.inf
        
        log_term = np.log(2 * n / self.delta_sample)
        ratio = (self.C_cov * log_term) / (n * self.pi_min)
        return self.L_LLR * (ratio ** (1.0 / self.d_joint))
    
    def compute_tail_error(self) -> float:
        """
        Compute tail error
        
        Δ_tail(σ_c^eff) = -log(1 - σ_c^eff)
        """
        return -np.log(1 - self.sigma_eff)
    
    def compute_bound(
        self,
        b: np.ndarray,
        fiber_samples: Optional[np.ndarray] = None,
        n: int = 100,
        proposal_distribution: Optional[Callable[[int], np.ndarray]] = None,
    ) -> float:
        """
        Compute primary safety bound U_prim(b)
        
        U_prim(b) = max_{y ∈ P̂_ε^N(b)} LLR_Y(y) + Δ_geom + Δ_sample + Δ_tail
        
        Args:
            b: Observed behavior
            fiber_samples: Pre-computed fiber samples (optional)
            n: Number of samples if fiber_samples not provided
            proposal_distribution: Proposal distribution if fiber_samples not provided
            
        Returns:
            Upper bound U_prim(b)
        """
        # Get fiber samples
        if fiber_samples is None:
            if proposal_distribution is None:
                # Default: sample from identity model
                proposal_distribution = lambda n_samples: self.identity_model.sample(
                    n_samples, z=self.identity, c=self.context
                )
            
            fiber_sampler = EpsilonSamplingFiber(
                scenario_mapping=self.scenario_mapping,
                proposal_distribution=proposal_distribution,
                epsilon=self.epsilon,
                pi_min=self.pi_min,
            )
            fiber_samples = fiber_sampler.sample(b, n)
        
        if len(fiber_samples) == 0:
            # No fiber samples found - return very conservative bound
            return np.inf
        
        # Compute max LLR_Y over fiber
        llr_values = [self.compute_llr_y(y) for y in fiber_samples]
        max_llr = np.max(llr_values)
        
        # Compute error terms
        delta_geom = self.compute_geometric_error()
        delta_sample = self.compute_sample_error(len(fiber_samples))
        delta_tail = self.compute_tail_error()
        
        return max_llr + delta_geom + delta_sample + delta_tail


def compute_primary_bound(
    b: np.ndarray,
    scenario_mapping: ScenarioMapping,
    identity_model: IdentityModel,
    background_model: BackgroundModel,
    identity: str,
    context: Optional[str] = None,
    n: int = 100,
    epsilon: float = 0.1,
    **kwargs
) -> float:
    """
    Convenience function for computing primary bound
    
    Args:
        b: Observed behavior
        scenario_mapping: Mapping G_c
        identity_model: P_θ
        background_model: P_u
        identity: Identity z
        context: Context c
        n: Number of samples
        epsilon: Geometric tolerance
        **kwargs: Additional parameters for SafetyBound
        
    Returns:
        Upper bound U_prim(b)
    """
    bound_computer = SafetyBound(
        scenario_mapping=scenario_mapping,
        identity_model=identity_model,
        background_model=background_model,
        identity=identity,
        context=context,
        epsilon=epsilon,
        **kwargs
    )
    return bound_computer.compute_bound(b, n=n)

