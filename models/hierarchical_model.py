"""
Hierarchical Dual-Space Behavior Model (§3.1)

Implements the four-component model M_c:
- (A) Hierarchical parameter space Y
- (B) Hierarchical behavior space B
- (C) Piecewise regular scenario mapping G_c
- (D) Generative priors and tail control
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class Layer:
    """Represents a layer in hierarchical space"""
    name: str
    center: np.ndarray
    radius: float
    dimension: int
    epsilon: float = 0.1
    
    def tube_neighborhood(self, y: np.ndarray) -> bool:
        """Check if y is in the tube neighborhood Y_r(epsilon)"""
        dist = np.linalg.norm(y - self.center)
        return dist <= self.radius + self.epsilon


class ScenarioMapping:
    """
    Piecewise regular scenario mapping G_c: Y -> B
    
    Properties:
    1. Bilipschitz within layers
    2. Good set regularity (differentiable, non-zero Jacobian)
    3. Finite multi-valuedness (bounded fiber cardinality)
    """
    
    def __init__(
        self,
        mapping_func: Callable[[np.ndarray], np.ndarray],
        jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lipschitz_const: float = 1.0,
        max_fiber_cardinality: int = 10,
    ):
        """
        Args:
            mapping_func: Function implementing G_c(y)
            jacobian_func: Function computing Jacobian determinant J_{c,r}(y)
            lipschitz_const: Lipschitz constant L_c
            max_fiber_cardinality: Upper bound M_c on fiber cardinality
        """
        self.mapping_func = mapping_func
        self.jacobian_func = jacobian_func
        self.lipschitz_const = lipschitz_const
        self.max_fiber_cardinality = max_fiber_cardinality
    
    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Apply mapping G_c(y)"""
        return self.mapping_func(y)
    
    def jacobian_det(self, y: np.ndarray) -> float:
        """Compute Jacobian determinant |det DG_c(y)|"""
        if self.jacobian_func is not None:
            return self.jacobian_func(y)
        else:
            # Numerical approximation
            return self._numerical_jacobian_det(y)
    
    def _numerical_jacobian_det(self, y: np.ndarray, eps: float = 1e-5) -> float:
        """Numerical approximation of Jacobian determinant"""
        n = len(y)
        jacobian = np.zeros((self.mapping_func(y).shape[0], n))
        
        for i in range(n):
            y_pert = y.copy()
            y_pert[i] += eps
            jacobian[:, i] = (self.mapping_func(y_pert) - self.mapping_func(y)) / eps
        
        return np.abs(np.linalg.det(jacobian))
    
    def inverse_fiber(self, b: np.ndarray, y_candidates: np.ndarray, 
                     epsilon: float = 0.1) -> List[np.ndarray]:
        """
        Find fiber G_c^{-1}(b) by checking candidates
        
        Args:
            b: Behavior point
            y_candidates: Candidate parameter points
            epsilon: Tolerance for fiber membership
            
        Returns:
            List of parameter points in fiber
        """
        fiber_points = []
        for y in y_candidates:
            b_pred = self.mapping_func(y)
            if np.linalg.norm(b_pred - b) <= epsilon:
                fiber_points.append(y)
        return fiber_points


class HierarchicalDualSpaceModel:
    """
    Hierarchical Dual-Space Behavior Model M_c
    
    M_c = ((Y, μ_Y), (B, μ_B), G_c, {P_θ, P_u})
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        y_layers: List[Layer],
        b_layers: List[Layer],
        sigma_eff: float = 0.02,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            y_layers: Layers in parameter space Y
            b_layers: Layers in behavior space B
            sigma_eff: Effective tail parameter σ_c^eff
        """
        self.scenario_mapping = scenario_mapping
        self.y_layers = y_layers
        self.b_layers = b_layers
        self.sigma_eff = sigma_eff
        
        # Dimension alignment check
        self._check_dimension_alignment()
    
    def _check_dimension_alignment(self):
        """Verify dimension alignment between Y and B layers"""
        for y_layer in self.y_layers:
            # Find corresponding B layer
            b_layer = self._find_corresponding_b_layer(y_layer)
            if b_layer is not None:
                assert y_layer.dimension == b_layer.dimension, \
                    f"Dimension mismatch: Y layer {y_layer.name} has dim {y_layer.dimension}, " \
                    f"but B layer {b_layer.name} has dim {b_layer.dimension}"
    
    def _find_corresponding_b_layer(self, y_layer: Layer) -> Optional[Layer]:
        """Find B layer corresponding to Y layer"""
        # Simple heuristic: find layer with matching dimension
        for b_layer in self.b_layers:
            if b_layer.dimension == y_layer.dimension:
                return b_layer
        return None
    
    def get_main_mass_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get main mass set Y_main = ∪_r Y_r(epsilon) and tail set Y_tail
        
        Returns:
            (main_set_mask, tail_set_mask) for a given set of points
        """
        # This is a placeholder - actual implementation depends on point set
        pass
    
    def classify_point(self, y: np.ndarray) -> Tuple[str, bool]:
        """
        Classify point as belonging to a layer or tail
        
        Returns:
            (layer_name, is_in_main_mass)
        """
        for layer in self.y_layers:
            if layer.tube_neighborhood(y):
                return (layer.name, True)
        return ("tail", False)
    
    def compute_tail_error(self) -> float:
        """Compute tail error Δ_tail = -log(1 - σ_c^eff)"""
        return -np.log(1 - self.sigma_eff)

