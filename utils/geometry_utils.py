"""
Geometry utility functions
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression


def compute_correlation_integral(points: np.ndarray, r: float) -> float:
    """
    Compute correlation integral C(r)
    
    C(r) = (2/(N(N-1))) Σ_{i<j} 1(||x_i - x_j|| < r)
    """
    if len(points) < 2:
        return 0.0
    
    distances = pdist(points)
    count = np.sum(distances < r)
    n = len(points)
    return 2 * count / (n * (n - 1))


def estimate_intrinsic_dimension(
    points: np.ndarray,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    n_radii: int = 50,
) -> tuple[float, float]:
    """
    Estimate intrinsic dimension using correlation integral
    
    Fits: log C(r) ≈ d * log r + C
    
    Returns:
        (estimated_dimension, R^2)
    """
    if len(points) < 2:
        return (0.0, 0.0)
    
    distances = pdist(points)
    
    if r_min is None:
        r_min = np.percentile(distances, 5)
    if r_max is None:
        r_max = np.percentile(distances, 95)
    
    if r_min >= r_max:
        return (0.0, 0.0)
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    correlation_integrals = []
    
    for r in radii:
        C_r = compute_correlation_integral(points, r)
        correlation_integrals.append(C_r)
    
    # Fit log-log relationship
    mask = np.array(correlation_integrals) > 0
    if np.sum(mask) < 3:
        return (0.0, 0.0)
    
    log_r = np.log(radii[mask])
    log_C = np.log(np.array(correlation_integrals)[mask])
    
    reg = LinearRegression()
    reg.fit(log_r.reshape(-1, 1), log_C)
    
    estimated_dim = reg.coef_[0]
    r_squared = reg.score(log_r.reshape(-1, 1), log_C)
    
    return (estimated_dim, r_squared)

