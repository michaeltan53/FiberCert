"""
FiberSamplingCert: Main Authentication System

Integrates all components into a complete authentication framework
"""

import numpy as np
from typing import Optional, Dict, Callable
from .models.hierarchical_model import HierarchicalDualSpaceModel, ScenarioMapping
from .models.generative_models import IdentityModel, BackgroundModel
from .fingerprint.fingerprint import BehaviorFingerprint
from .fiber_sampling.safety_bounds import SafetyBound
from .fiber_sampling.fiber_sampling import EpsilonSamplingFiber
from .monitoring.regime_switching import RegimeSwitcher, RegimeState
from .monitoring.monitoring import MonitoringSystem, create_default_monitoring_features
from .monitoring.base_bound import BaseBound


class FiberSamplingCert:
    """
    Main FiberSamplingCert authentication system
    
    Integrates:
    - Hierarchical dual-space model
    - Behavior fingerprints
    - Fiber sampling and safety bounds
    - Monitoring and regime switching
    """
    
    def __init__(
        self,
        hierarchical_model: HierarchicalDualSpaceModel,
        identity_model: IdentityModel,
        background_model: BackgroundModel,
        identity: str,
        context: Optional[str] = None,
        epsilon: float = 0.1,
        n_samples: int = 100,
        enable_monitoring: bool = True,
        **kwargs
    ):
        """
        Args:
            hierarchical_model: Hierarchical dual-space model M_c
            identity_model: Identity model P_θ
            background_model: Background model P_u
            identity: Identity z to authenticate
            context: Context c
            epsilon: Geometric tolerance
            n_samples: Number of fiber samples
            enable_monitoring: Enable monitoring and regime switching
            **kwargs: Additional parameters for safety bounds
        """
        self.hierarchical_model = hierarchical_model
        self.identity_model = identity_model
        self.background_model = background_model
        self.identity = identity
        self.context = context
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.enable_monitoring = enable_monitoring
        
        # Create fingerprints
        scenario_mapping = hierarchical_model.scenario_mapping
        self.fingerprint_z = BehaviorFingerprint(
            scenario_mapping=scenario_mapping,
            generative_model=identity_model,
            identity=identity,
            context=context,
        )
        self.fingerprint_u = BehaviorFingerprint(
            scenario_mapping=scenario_mapping,
            generative_model=background_model,
            identity=None,
            context=context,
        )
        
        # Create safety bound computer
        self.safety_bound = SafetyBound(
            scenario_mapping=scenario_mapping,
            identity_model=identity_model,
            background_model=background_model,
            identity=identity,
            context=context,
            epsilon=epsilon,
            **kwargs
        )
        
        # Create base bound
        self.base_bound = BaseBound(
            fingerprint_z=self.fingerprint_z,
            fingerprint_u=self.fingerprint_u,
            delta_base=kwargs.get("delta_base", 0.01),
        )
        
        # Create monitoring and regime switching
        if enable_monitoring:
            # Create fiber sampler for monitoring
            def proposal_dist(n: int):
                return identity_model.sample(n, z=identity, c=context)
            
            fiber_sampler = EpsilonSamplingFiber(
                scenario_mapping=scenario_mapping,
                proposal_distribution=proposal_dist,
                epsilon=epsilon,
            )
            
            # Create monitoring features
            def compute_llr_y(y):
                log_p_theta = identity_model.log_prob(y, z=identity, c=context)
                log_p_u = background_model.log_prob(y, z=None, c=context)
                if log_p_u == -np.inf:
                    return np.inf
                return log_p_theta - log_p_u
            
            monitoring_features = create_default_monitoring_features(
                scenario_mapping=scenario_mapping,
                fiber_sampler=fiber_sampler,
                compute_llr_y=compute_llr_y,
            )
            
            monitoring_system = MonitoringSystem(
                features=monitoring_features,
                window_size=kwargs.get("window_size", 100),
            )
            
            # Create regime switcher
            def compute_primary_bound(b):
                return self.safety_bound.compute_bound(b, n=n_samples)
            
            def compute_base_bound(b):
                return self.base_bound.compute_bound(b)
            
            self.regime_switcher = RegimeSwitcher(
                monitoring_system=monitoring_system,
                compute_primary_bound=compute_primary_bound,
                compute_base_bound=compute_base_bound,
            )
        else:
            self.regime_switcher = None
    
    def authenticate(
        self,
        b: np.ndarray,
        return_details: bool = False,
    ) -> Dict:
        """
        Authenticate behavior observation
        
        Args:
            b: Behavior observation
            return_details: Return detailed information
            
        Returns:
            Authentication result dictionary
        """
        # Prepare context for monitoring
        context = {}
        
        if self.enable_monitoring and self.regime_switcher is not None:
            # Sample fiber for monitoring
            def proposal_dist(n: int):
                return self.identity_model.sample(n, z=self.identity, c=self.context)
            
            fiber_sampler = EpsilonSamplingFiber(
                scenario_mapping=self.hierarchical_model.scenario_mapping,
                proposal_distribution=proposal_dist,
                epsilon=self.epsilon,
            )
            
            fiber_samples = fiber_sampler.sample(b, n=self.n_samples)
            context["fiber_samples"] = fiber_samples
            
            # Compute bounds with different sample sizes for monitoring
            bound_N = self.safety_bound.compute_bound(b, fiber_samples=fiber_samples, n=self.n_samples)
            bound_N4 = self.safety_bound.compute_bound(b, fiber_samples=fiber_samples[:len(fiber_samples)//4], n=self.n_samples//4)
            context["bound_N"] = bound_N
            context["bound_N4"] = bound_N4
        
        # Update regime and compute bound
        if self.enable_monitoring and self.regime_switcher is not None:
            self.regime_switcher.update(b, context)
            bound = self.regime_switcher.compute_bound(b)
            regime = self.regime_switcher.get_state()
        else:
            # No monitoring: always use primary bound
            bound = self.safety_bound.compute_bound(b, n=self.n_samples)
            regime = RegimeState.A
        
        # Compute LLR for decision
        llr = self.fingerprint_z.log_density(b) - self.fingerprint_u.log_density(b)
        
        # Authentication decision (simplified: bound > threshold means attack)
        threshold = 0.0  # Can be calibrated
        is_authenticated = bound < threshold
        
        result = {
            "is_authenticated": is_authenticated,
            "bound": bound,
            "llr": llr,
            "regime": regime.value if isinstance(regime, RegimeState) else regime,
        }
        
        if return_details:
            result["details"] = {
                "fiber_samples": context.get("fiber_samples", None),
                "bound_N": context.get("bound_N", None),
                "bound_N4": context.get("bound_N4", None),
            }
        
        return result
    
    def get_availability(self) -> float:
        """Get system availability (fraction of time in Regime A)"""
        if self.regime_switcher is not None:
            return self.regime_switcher.get_availability()
        return 1.0  # Always available if no monitoring


if __name__ == "__main__":
    # Example usage
    print("FiberSamplingCert Main System")
    print("See experiments/ for usage examples")

