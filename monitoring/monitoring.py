"""
Monitoring Features (§3.4)

Lightweight monitoring features {g_j} for detecting geometric drift
"""

import numpy as np
from typing import List, Callable, Optional
from collections import deque
from dataclasses import dataclass
from ..models.hierarchical_model import ScenarioMapping
from ..fiber_sampling.fiber_sampling import EpsilonSamplingFiber


@dataclass
class MonitoringFeature:
    """Single monitoring feature"""
    name: str
    compute_func: Callable[[np.ndarray, dict], float]
    threshold: float
    alpha_j: float = 0.01  # Individual false alarm rate


class MonitoringSystem:
    """
    Monitoring system for detecting geometric drift
    
    Maintains sliding window statistics and triggers regime switching
    """
    
    def __init__(
        self,
        features: List[MonitoringFeature],
        window_size: int = 100,
        alpha_global: Optional[float] = None,
    ):
        """
        Args:
            features: List of monitoring features {g_j}
            window_size: Sliding window length W
            alpha_global: Global false alarm rate (auto-computed if None)
        """
        self.features = features
        self.window_size = window_size
        self.windows: dict = {f.name: deque(maxlen=window_size) for f in features}
        
        if alpha_global is None:
            # Sum of individual false alarm rates
            alpha_global = sum(f.alpha_j for f in features)
        self.alpha_global = alpha_global
    
    def update(self, b: np.ndarray, context: dict):
        """
        Update monitoring with new behavior observation
        
        Args:
            b: New behavior observation
            context: Context dictionary (fiber samples, bounds, etc.)
        """
        for feature in self.features:
            value = feature.compute_func(b, context)
            self.windows[feature.name].append(value)
    
    def get_window_statistics(self, feature_name: str) -> dict:
        """Get window statistics for a feature"""
        values = list(self.windows[feature_name])
        if len(values) == 0:
            return {"mean": 0.0, "max": 0.0, "min": 0.0}
        
        return {
            "mean": np.mean(values),
            "max": np.max(values),
            "min": np.min(values),
            "std": np.std(values),
        }
    
    def check_drift(self) -> bool:
        """
        Check if any feature indicates drift
        
        Uses conservative "OR" logic: any feature exceeding threshold triggers drift
        
        Returns:
            True if drift detected
        """
        for feature in self.features:
            stats = self.get_window_statistics(feature.name)
            if stats["mean"] > feature.threshold:
                return True
        return False
    
    def get_drift_features(self) -> List[str]:
        """Get list of features that indicate drift"""
        drifted = []
        for feature in self.features:
            stats = self.get_window_statistics(feature.name)
            if stats["mean"] > feature.threshold:
                drifted.append(feature.name)
        return drifted


def create_default_monitoring_features(
    scenario_mapping: ScenarioMapping,
    fiber_sampler: Optional[EpsilonSamplingFiber] = None,
    compute_llr_y: Optional[Callable] = None,
) -> List[MonitoringFeature]:
    """
    Create default monitoring features from §5.3.2
    
    Features:
    1. g_1: Local fiber dimension (Z-score)
    2. g_2: Sampling spread (max eigenvalue of covariance)
    3. g_3: Bound conservatism residual (difference between N and N/4 bounds)
    4. g_4: Trajectory local smoothness (L2 norm of acceleration changes)
    """
    features = []
    
    # Feature 1: Local fiber dimension
    def compute_fiber_dimension(b: np.ndarray, context: dict) -> float:
        """Compute Z-score of fiber dimension"""
        if "fiber_samples" not in context or len(context["fiber_samples"]) < 10:
            return 0.0
        
        y_fiber = context["fiber_samples"]
        
        # Estimate dimension (simplified)
        from scipy.spatial.distance import pdist
        distances = pdist(y_fiber)
        if len(distances) == 0:
            return 0.0
        
        # Use correlation dimension estimate
        r = np.percentile(distances, 50)
        count = np.sum(distances < r)
        if count == 0:
            return 0.0
        
        # Simplified dimension estimate
        n = len(y_fiber)
        est_dim = np.log(count) / np.log(n) if n > 1 else 0.0
        
        # Z-score (assuming expected dimension ~2-4)
        expected_dim = 3.0
        std_dim = 1.0
        z_score = abs(est_dim - expected_dim) / std_dim
        
        return z_score
    
    features.append(MonitoringFeature(
        name="fiber_dimension",
        compute_func=compute_fiber_dimension,
        threshold=2.0,  # 2-sigma threshold
        alpha_j=0.003,
    ))
    
    # Feature 2: Sampling spread
    def compute_sampling_spread(b: np.ndarray, context: dict) -> float:
        """Compute max eigenvalue of fiber covariance"""
        if "fiber_samples" not in context or len(context["fiber_samples"]) < 2:
            return 0.0
        
        y_fiber = context["fiber_samples"]
        cov = np.cov(y_fiber.T)
        if cov.ndim == 0:
            return 0.0
        
        eigenvals = np.linalg.eigvals(cov)
        return np.max(eigenvals)
    
    features.append(MonitoringFeature(
        name="sampling_spread",
        compute_func=compute_sampling_spread,
        threshold=1.0,  # Threshold depends on scale
        alpha_j=0.003,
    ))
    
    # Feature 3: Bound conservatism residual
    def compute_bound_residual(b: np.ndarray, context: dict) -> float:
        """Compute difference between N and N/4 bounds"""
        if "bound_N" not in context or "bound_N4" not in context:
            return 0.0
        
        return abs(context["bound_N"] - context["bound_N4"])
    
    features.append(MonitoringFeature(
        name="bound_residual",
        compute_func=compute_bound_residual,
        threshold=0.5,  # Threshold for bound difference
        alpha_j=0.002,
    ))
    
    # Feature 4: Trajectory smoothness
    def compute_trajectory_smoothness(b: np.ndarray, context: dict) -> float:
        """Compute L2 norm of acceleration changes"""
        if "trajectory" not in context or len(context["trajectory"]) < 3:
            return 0.0
        
        traj = context["trajectory"]
        if len(traj) < 3:
            return 0.0
        
        # Compute second differences (acceleration changes)
        accel_changes = np.diff(traj, n=2, axis=0)
        return np.linalg.norm(accel_changes)
    
    features.append(MonitoringFeature(
        name="trajectory_smoothness",
        compute_func=compute_trajectory_smoothness,
        threshold=10.0,  # Threshold depends on trajectory scale
        alpha_j=0.002,
    ))
    
    return features

