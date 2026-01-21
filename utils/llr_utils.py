"""
LLR computation utilities
"""

import numpy as np
from typing import Optional
from ..models.generative_models import IdentityModel, BackgroundModel


def compute_llr_y(
    y: np.ndarray,
    identity_model: IdentityModel,
    background_model: BackgroundModel,
    identity: str,
    context: Optional[str] = None,
) -> float:
    """
    Compute parameter space LLR
    
    LLR_Y(y) = log(P_θ(y|z,c) / P_u(y|c))
    """
    log_p_theta = identity_model.log_prob(y, z=identity, c=context)
    log_p_u = background_model.log_prob(y, z=None, c=context)
    
    if log_p_u == -np.inf:
        return np.inf
    
    return log_p_theta - log_p_u


def compute_llr_b_true(
    b: np.ndarray,
    scenario_mapping,
    identity_model: IdentityModel,
    background_model: BackgroundModel,
    identity: str,
    context: Optional[str] = None,
    grid_resolution: int = 100,
    epsilon: float = 0.1,
) -> float:
    """
    Compute "true" behavior space LLR using grid enumeration (Toy Model)
    
    LLR_B^true(b) = log(Σ_{y∈G∩G_c^{-1}(b)} p_θ(y)w(y) / Σ_{y∈G∩G_c^{-1}(b)} p_u(y)w(y))
    
    Args:
        b: Behavior point
        scenario_mapping: Mapping G_c
        identity_model: P_θ
        background_model: P_u
        identity: Identity z
        context: Context c
        grid_resolution: Grid resolution for enumeration
        epsilon: Tolerance for fiber membership
        
    Returns:
        True LLR_B(b)
    """
    # Create grid in parameter space (assuming 2D for Toy Model)
    y1 = np.linspace(-1, 1, grid_resolution)
    y2 = np.linspace(-1, 1, grid_resolution)
    Y1, Y2 = np.meshgrid(y1, y2)
    y_grid = np.column_stack([Y1.ravel(), Y2.ravel()])
    
    # Find fiber points
    fiber_points = []
    fiber_weights = []
    
    for y in y_grid:
        b_pred = scenario_mapping(y)
        if np.linalg.norm(b_pred - b) <= epsilon:
            fiber_points.append(y)
            # Compute weight (Jacobian determinant)
            jacobian_det = scenario_mapping.jacobian_det(y)
            if jacobian_det > 1e-10:
                fiber_weights.append(jacobian_det)
            else:
                fiber_weights.append(0.0)
    
    if len(fiber_points) == 0:
        return 0.0
    
    fiber_points = np.array(fiber_points)
    fiber_weights = np.array(fiber_weights)
    
    # Compute weighted sums
    sum_theta = 0.0
    sum_u = 0.0
    
    for i, y in enumerate(fiber_points):
        w = fiber_weights[i]
        if w > 0:
            log_p_theta = identity_model.log_prob(y, z=identity, c=context)
            log_p_u = background_model.log_prob(y, z=None, c=context)
            
            sum_theta += np.exp(log_p_theta) * w
            sum_u += np.exp(log_p_u) * w
    
    if sum_u == 0:
        return np.inf if sum_theta > 0 else 0.0
    
    return np.log(sum_theta / sum_u) if sum_theta > 0 else -np.inf

