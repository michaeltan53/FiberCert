"""
Regime Switching (§3.4)

Dual-state (A/B) decision system with monitoring-triggered switching
"""

import numpy as np
from enum import Enum
from typing import Optional, Callable
from .monitoring import MonitoringSystem
from ..fiber_sampling.safety_bounds import SafetyBound


class RegimeState(Enum):
    """Regime states"""
    A = "A"  # Structured authentication (efficient)
    B = "B"  # Conservative authentication (safe fallback)


class RegimeSwitcher:
    """
    Regime switching system
    
    State A: Use U_prim(b) (efficient, requires geometric assumptions)
    State B: Use U_base(b) (conservative, no geometric assumptions)
    """
    
    def __init__(
        self,
        monitoring_system: MonitoringSystem,
        compute_primary_bound: Callable[[np.ndarray], float],
        compute_base_bound: Callable[[np.ndarray], float],
        initial_state: RegimeState = RegimeState.A,
    ):
        """
        Args:
            monitoring_system: Monitoring system for drift detection
            compute_primary_bound: Function to compute U_prim(b)
            compute_base_bound: Function to compute U_base(b)
            initial_state: Initial regime state
        """
        self.monitoring_system = monitoring_system
        self.compute_primary_bound = compute_primary_bound
        self.compute_base_bound = compute_base_bound
        self.current_state = initial_state
        self.state_history = [initial_state]
    
    def update(self, b: np.ndarray, context: dict) -> RegimeState:
        """
        Update monitoring and potentially switch regime
        
        Args:
            b: New behavior observation
            context: Context dictionary for monitoring
            
        Returns:
            Current regime state
        """
        # Update monitoring
        self.monitoring_system.update(b, context)
        
        # Check for drift
        if self.monitoring_system.check_drift():
            # Switch to conservative Regime B
            self.current_state = RegimeState.B
        else:
            # Can use efficient Regime A
            self.current_state = RegimeState.A
        
        self.state_history.append(self.current_state)
        return self.current_state
    
    def compute_bound(self, b: np.ndarray) -> float:
        """
        Compute authentication bound based on current regime
        
        U(b,t) = {
            U_prim(b),  if S(t) = A
            U_base(b),  if S(t) = B
        }
        
        Args:
            b: Behavior observation
            
        Returns:
            Upper bound U(b,t)
        """
        if self.current_state == RegimeState.A:
            return self.compute_primary_bound(b)
        else:
            return self.compute_base_bound(b)
    
    def get_state(self) -> RegimeState:
        """Get current regime state"""
        return self.current_state
    
    def get_availability(self) -> float:
        """
        Compute availability: fraction of time in Regime A
        
        Returns:
            Availability (0-1)
        """
        if len(self.state_history) == 0:
            return 0.0
        
        count_A = sum(1 for s in self.state_history if s == RegimeState.A)
        return count_A / len(self.state_history)

