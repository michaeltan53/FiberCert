"""
Base Bound U_base (§3.4)

Conservative bound that doesn't rely on geometric assumptions
"""

import numpy as np
from typing import Optional, Callable, Tuple
from ..models.generative_models import IdentityModel, BackgroundModel
from ..models.hierarchical_model import ScenarioMapping
from ..fingerprint.fingerprint import compute_llr_behavior_space, BehaviorFingerprint


class BaseBound:
    """
    Conservative base bound U_base(b)
    
    Doesn't rely on geometric assumptions, only requires:
    Pr(LLR_B(b) > U_base(b)) ≤ δ_base
    """
    
    def __init__(
        self,
        fingerprint_z: BehaviorFingerprint,
        fingerprint_u: BehaviorFingerprint,
        delta_base: float = 0.01,
        calibration_data: Optional[np.ndarray] = None,
    ):
        """
        Args:
            fingerprint_z: F_{z,c} fingerprint
            fingerprint_u: F_{u,c} fingerprint
            delta_base: Failure probability δ_base
            calibration_data: Calibration data for threshold setting
        """
        self.fingerprint_z = fingerprint_z
        self.fingerprint_u = fingerprint_u
        self.delta_base = delta_base
        self.calibration_data = calibration_data
        
        # Calibrate threshold if data provided
        self.threshold = self._calibrate_threshold() if calibration_data is not None else 0.0
    
    def _calibrate_threshold(self) -> float:
        """
        Calibrate threshold to satisfy coverage requirement
        
        Find threshold such that:
        Pr(LLR_B(b) > threshold) ≤ δ_base
        """
        if self.calibration_data is None:
            return 0.0
        
        # Compute LLR for calibration data
        llr_values = []
        for b in self.calibration_data:
            llr = compute_llr_behavior_space(
                b, self.fingerprint_z, self.fingerprint_u
            )
            llr_values.append(llr)
        
        llr_values = np.array(llr_values)
        
        # Find threshold at (1 - δ_base) quantile
        threshold = np.percentile(llr_values, (1 - self.delta_base) * 100)
        
        return threshold
    
    def compute_bound(self, b: np.ndarray) -> float:
        """
        Compute conservative base bound
        
        U_base(b) = threshold (constant) or LLR_B(b) + margin
        
        Args:
            b: Behavior observation
            
        Returns:
            Upper bound U_base(b)
        """
        # Option 1: Constant threshold (simplest)
        if self.threshold > 0:
            return self.threshold
        
        # Option 2: LLR + margin (more adaptive)
        llr = compute_llr_behavior_space(
            b, self.fingerprint_z, self.fingerprint_u
        )
        
        # Add conservative margin
        margin = 2.0  # Conservative margin
        return llr + margin
    
    def compute_with_far_constraint(
        self,
        b: np.ndarray,
        alpha_base: float = 0.01,
    ) -> Tuple[float, bool]:
        """
        Compute bound with FAR constraint
        
        Args:
            b: Behavior observation
            alpha_base: Target false alarm rate
            
        Returns:
            (bound, is_authenticated)
        """
        bound = self.compute_bound(b)
        
        # Authentication decision: LLR > threshold means reject (attack)
        # For conservative approach, use bound directly
        is_authenticated = bound < self.threshold if self.threshold > 0 else bound < 0
        
        return (bound, is_authenticated)

