"""
Geometry Regularity Checking (§3.3.1, §5.1)

Assumption 3.1: Ideal geometric regularity
- Ahlfors regular structure
- Local Lipschitz regularity
- Tail error control
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from ..models.hierarchical_model import ScenarioMapping


class AhlforsRegularityChecker:
    """
    Check Ahlfors regularity of joint mapping image set
    
    S_b = Φ_c(B_R(y_0) ∩ Y_geo) is d_joint-dimensional Ahlfors regular
    """
    
    def __init__(self, d_joint: float, C_A: float = 1.0):
        """
        Args:
            d_joint: Joint dimension d_joint
            C_A: Ahlfors regularity constant C_A
        """
        self.d_joint = d_joint
        self.C_A = C_A
    
    def estimate_dimension(
        self,
        points: np.ndarray,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Estimate intrinsic dimension using correlation integral
        
        C(r) = (2/(N(N-1))) Σ_{i<j} 1(||x_i - x_j|| < r)
        log C(r) ≈ d_joint * log r + C
        
        Args:
            points: Point cloud
            r_min: Minimum radius (auto if None)
            r_max: Maximum radius (auto if None)
            
        Returns:
            (estimated_dimension, R^2)
        """
        if len(points) < 2:
            return 0.0, 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Auto-select radius range
        if r_min is None:
            r_min = np.percentile(distances, 5)
        if r_max is None:
            r_max = np.percentile(distances, 95)
        
        # Compute correlation integral for different radii
        n_radii = 50
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
        correlation_integrals = []
        
        for r in radii:
            count = np.sum(distances < r)
            C_r = 2 * count / (len(points) * (len(points) - 1))
            correlation_integrals.append(C_r)
        
        # Fit log-log linear relationship
        mask = np.array(correlation_integrals) > 0
        if np.sum(mask) < 3:
            return (0.0, 0.0)
        
        log_r = np.log(radii[mask])
        log_C = np.log(np.array(correlation_integrals)[mask])
        
        # Linear regression
        reg = LinearRegression()
        reg.fit(log_r.reshape(-1, 1), log_C)
        
        estimated_dim = reg.coef_[0]
        r_squared = reg.score(log_r.reshape(-1, 1), log_C)
        
        return (estimated_dim, r_squared)
    
    def check_regularity(
        self,
        points: np.ndarray,
        threshold_r2: float = 0.9,
        threshold_C_A: float = 10.0,
    ) -> Tuple[bool, float, float]:
        """
        Check if point set is Ahlfors regular
        
        Args:
            points: Point cloud
            threshold_r2: Minimum R^2 for good fit
            threshold_C_A: Maximum C_A for regularity
            
        Returns:
            (is_regular, estimated_dim, R^2)
        """
        est_dim, r_squared = self.estimate_dimension(points)
        
        # Estimate C_A (simplified)
        if len(points) > 0:
            # Use volume growth as proxy
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            r_max = np.max(distances)
            r_min = np.percentile(distances, 10)
            
            # Approximate volume growth
            volume_ratio = (r_max / r_min) ** self.d_joint if r_min > 0 else np.inf
            C_A_est = volume_ratio / len(points) if len(points) > 0 else np.inf
        else:
            C_A_est = np.inf
        
        is_regular = (
            r_squared >= threshold_r2 and
            C_A_est <= threshold_C_A and
            abs(est_dim - self.d_joint) < 0.5 * self.d_joint
        )
        
        return is_regular, est_dim, r_squared


class GeometryRegularityChecker:
    """
    Check geometric regularity assumptions for joint mapping
    
    Implements checks from §5.1.2
    """
    
    def __init__(
        self,
        scenario_mapping: ScenarioMapping,
        d_joint: float,
        compute_llr_y: Optional[callable] = None,
    ):
        """
        Args:
            scenario_mapping: Mapping G_c
            d_joint: Expected joint dimension
            compute_llr_y: Function to compute LLR_Y(y)
        """
        self.scenario_mapping = scenario_mapping
        self.d_joint = d_joint
        self.compute_llr_y = compute_llr_y
        self.ahlfors_checker = AhlforsRegularityChecker(d_joint)
    
    def check_point(
        self,
        b: np.ndarray,
        y_fiber: np.ndarray,
        threshold_r2: float = 0.9,
        threshold_C_A: float = 10.0,
    ) -> Tuple[bool, dict]:
        """
        Check geometric regularity for a behavior point
        
        Args:
            b: Behavior point
            y_fiber: Fiber samples
            threshold_r2: R^2 threshold
            threshold_C_A: C_A threshold
            
        Returns:
            (is_good, metrics_dict)
        """
        if len(y_fiber) < 10:
            return False, {"reason": "insufficient_samples"}
        
        # Construct joint embedding
        joint_points = []
        for y in y_fiber:
            b_pred = self.scenario_mapping(y)
            if self.compute_llr_y is not None:
                llr = self.compute_llr_y(y)
                joint_point = np.concatenate([y, b_pred, llr])
                # joint_point = np.concatenate([
                #     np.ravel(y),
                #     np.ravel(b_pred),
                #     np.ravel(llr)])

            else:
                joint_point = np.concatenate([y, b_pred])
            joint_points.append(joint_point)
        
        joint_points = np.array(joint_points)
        
        # Check Ahlfors regularity
        is_regular, est_dim, r_squared = self.ahlfors_checker.check_regularity(
            joint_points, threshold_r2, threshold_C_A
        )
        
        # Check dimension
        dim_check = abs(est_dim - self.d_joint) < 0.5 * self.d_joint

        # Check if point is "bad"
        is_bad = (
            r_squared < threshold_r2 or
            not dim_check or
            est_dim > self.d_joint + 2.0  # Dimension inflation
        )
        
        metrics = {
            "estimated_dimension": est_dim,
            "r_squared": r_squared,
            "is_regular": is_regular,
            "dimension_check": dim_check,
            "is_bad": is_bad,
        }
        
        return not is_bad, metrics

