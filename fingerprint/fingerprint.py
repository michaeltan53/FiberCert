"""
Behavior Probability Fingerprints and LLR (§3.2)

Definition 4.1: Behavior Probability Fingerprint
F_{z,c} = (G_c)_# P_θ(·|z,c)
F_{u,c} = (G_c)_# P_u(·|c)

LLR_B(b) = log(f_{z,c}(b) / f_{u,c}(b))
"""

import numpy as np
from typing import Optional, List
from ..models.hierarchical_model import ScenarioMapping
from ..models.generative_models import IdentityModel, BackgroundModel
from .finite_branch_density import FiniteBranchDensity


class BehaviorFingerprint:
    """
    Behavior Probability Fingerprint
    
    Represents the pushforward measure F_{z,c} or F_{u,c}
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        generative_model: BackgroundModel,
        # generative_model: IdentityModel | BackgroundModel,
        identity: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            generative_model: Either P_θ or P_u
            identity: Identity z (required for P_θ)
            context: Context c
        """
        self.scenario_mapping = scenario_mapping
        self.generative_model = generative_model
        self.identity = identity
        self.context = context
    
    def log_density(self, b: np.ndarray, y_candidates: Optional[np.ndarray] = None,
                   epsilon: float = 0.1) -> float:
        """
        Compute log density f_{z,c}(b) or f_{u,c}(b) using finite branch formula
        
        f(b) = Σ_{y ∈ G_c^{-1}(b) ∩ Y_main} p(y) / J_{c,r(y)}(y)
        
        Args:
            b: Behavior point
            y_candidates: Candidate parameter points for fiber search
            epsilon: Tolerance for fiber membership
            
        Returns:
            log f(b)
        """
        # Find fiber
        if y_candidates is None:
            # Sample candidates from generative model
            y_candidates = self.generative_model.sample(
                n=1000, z=self.identity, c=self.context
            )
        
        fiber_points = self.scenario_mapping.inverse_fiber(b, y_candidates, epsilon)
        
        if len(fiber_points) == 0:
            # No fiber points found - return very negative value
            return -np.inf
        
        # Compute finite branch density
        density = 0.0
        for y in fiber_points:
            # Get probability p(y)
            log_p = self.generative_model.log_prob(
                y, z=self.identity, c=self.context
            )
            
            # Get Jacobian determinant
            jacobian_det = self.scenario_mapping.jacobian_det(y)
            
            if jacobian_det > 1e-10:  # Avoid division by zero
                density += np.exp(log_p) / jacobian_det
        
        return np.log(density) if density > 0 else -np.inf
    
    def density(self, b: np.ndarray, y_candidates: Optional[np.ndarray] = None,
               epsilon: float = 0.1) -> float:
        """Compute density f(b)"""
        return np.exp(self.log_density(b, y_candidates, epsilon))


def compute_llr_behavior_space(
    b: np.ndarray,
    fingerprint_z: BehaviorFingerprint,
    fingerprint_u: BehaviorFingerprint,
    y_candidates: Optional[np.ndarray] = None,
    epsilon: float = 0.1,
) -> float:
    """
    Compute behavior space log-likelihood ratio
    
    LLR_B(b) = log(f_{z,c}(b) / f_{u,c}(b))
    
    Args:
        b: Behavior point
        fingerprint_z: F_{z,c} fingerprint
        fingerprint_u: F_{u,c} fingerprint
        y_candidates: Candidate parameter points
        epsilon: Tolerance for fiber membership
        
    Returns:
        LLR_B(b)
    """
    log_f_z = fingerprint_z.log_density(b, y_candidates, epsilon)
    log_f_u = fingerprint_u.log_density(b, y_candidates, epsilon)
    
    if log_f_z == -np.inf and log_f_u == -np.inf:
        return 0.0  # Both densities are zero
    
    if log_f_u == -np.inf:
        return np.inf  # f_u is zero but f_z is not
    
    return log_f_z - log_f_u

