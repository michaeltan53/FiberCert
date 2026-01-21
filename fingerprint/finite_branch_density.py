"""
Finite Branch Density Formula (§3.2)

f_{z,c}(b) = Σ_{y ∈ G_c^{-1}(b) ∩ Y_main} p_θ(y|z,c) / J_{c,r(y)}(y)
f_{u,c}(b) = Σ_{y ∈ G_c^{-1}(b) ∩ Y_main} p_u(y|c) / J_{c,r(y)}(y)
"""

import numpy as np
from typing import List, Optional
from ..models.hierarchical_model import ScenarioMapping, HierarchicalDualSpaceModel
from ..models.generative_models import GenerativeModel


class FiniteBranchDensity:
    """
    Computes finite branch density using the formula from §3.2
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        hierarchical_model: Optional[HierarchicalDualSpaceModel] = None,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            hierarchical_model: Optional hierarchical model for layer classification
        """
        self.scenario_mapping = scenario_mapping
        self.hierarchical_model = hierarchical_model
    
    def compute_density(
        self,
        b: np.ndarray,
        generative_model: GenerativeModel,
        y_candidates: np.ndarray,
        epsilon: float = 0.1,
        identity: Optional[str] = None,
        context: Optional[str] = None,
    ) -> float:
        """
        Compute finite branch density
        
        Args:
            b: Behavior point
            generative_model: P_θ or P_u
            y_candidates: Candidate parameter points
            epsilon: Tolerance for fiber membership
            identity: Identity z (for P_θ)
            context: Context c
            
        Returns:
            Density f(b)
        """
        # Find fiber points
        fiber_points = self.scenario_mapping.inverse_fiber(b, y_candidates, epsilon)
        
        if len(fiber_points) == 0:
            return 0.0
        
        # Filter to main mass set if hierarchical model available
        if self.hierarchical_model is not None:
            fiber_points = [
                y for y in fiber_points
                if self.hierarchical_model.classify_point(y)[1]  # is_in_main_mass
            ]
        
        if len(fiber_points) == 0:
            return 0.0
        
        # Compute finite branch sum
        density = 0.0
        for y in fiber_points:
            # Get probability
            log_p = generative_model.log_prob(y, z=identity, c=context)
            
            # Get Jacobian determinant
            jacobian_det = self.scenario_mapping.jacobian_det(y)
            
            if jacobian_det > 1e-10:
                density += np.exp(log_p) / jacobian_det
        
        return density
    
    def compute_log_density(
        self,
        b: np.ndarray,
        generative_model: GenerativeModel,
        y_candidates: np.ndarray,
        epsilon: float = 0.1,
        identity: Optional[str] = None,
        context: Optional[str] = None,
    ) -> float:
        """Compute log density"""
        density = self.compute_density(
            b, generative_model, y_candidates, epsilon, identity, context
        )
        return np.log(density) if density > 0 else -np.inf

