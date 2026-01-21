"""
Generative Models P_θ and P_u (§3.1, §3.2)

P_θ(·|z,c): Identity-specific generative model
P_u(·|c): Background/universal generative model
"""

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GenerativeModel(ABC):
    """Base class for generative models"""
    
    @abstractmethod
    def log_prob(self, y: np.ndarray, z: Optional[str] = None, 
                c: Optional[str] = None) -> np.ndarray:
        """Compute log probability log P(y | z, c)"""
        pass
    
    @abstractmethod
    def sample(self, n: int, z: Optional[str] = None, 
              c: Optional[str] = None) -> np.ndarray:
        """Sample n points from the distribution"""
        pass
    
    def prob(self, y: np.ndarray, z: Optional[str] = None, 
            c: Optional[str] = None) -> np.ndarray:
        """Compute probability P(y | z, c)"""
        return np.exp(self.log_prob(y, z, c))


class IdentityModel(GenerativeModel):
    """
    Identity-specific model P_θ(·|z,c)
    
    Can be implemented as:
    - Parametric Gaussian mixture
    - Normalizing flow
    - Neural network density estimator
    """
    
    def __init__(
        self,
        model_type: str = "gmm",
        n_components: int = 5,
        **kwargs
    ):
        """
        Args:
            model_type: Type of model ("gmm", "flow", "nn")
            n_components: Number of components for GMM
        """
        self.model_type = model_type
        self.n_components = n_components
        self.models: Dict[str, Any] = {}  # Store per-identity models
        self.kwargs = kwargs
    
    def fit(self, y_data: np.ndarray, z: str, c: Optional[str] = None):
        """Fit model for identity z in context c"""
        key = f"{z}_{c}" if c else z
        
        if self.model_type == "gmm":
            model = GaussianMixture(
                n_components=self.n_components,
                **self.kwargs
            )
            model.fit(y_data)
            self.models[key] = model
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
    
    def log_prob(self, y: np.ndarray, z: Optional[str] = None, 
                c: Optional[str] = None) -> np.ndarray:
        """Compute log P_θ(y | z, c)"""
        if z is None:
            raise ValueError("Identity z must be provided")
        
        key = f"{z}_{c}" if c else z
        
        if key not in self.models:
            raise ValueError(f"Model for identity {z} not fitted")
        
        model = self.models[key]
        
        if self.model_type == "gmm":
            return model.score_samples(y.reshape(1, -1) if y.ndim == 1 else y)
        else:
            raise NotImplementedError
    
    def sample(self, n: int, z: Optional[str] = None, 
              c: Optional[str] = None) -> np.ndarray:
        """Sample from P_θ(·|z,c)"""
        if z is None:
            raise ValueError("Identity z must be provided")
        
        key = f"{z}_{c}" if c else z
        
        if key not in self.models:
            raise ValueError(f"Model for identity {z} not fitted")
        
        model = self.models[key]
        
        if self.model_type == "gmm":
            return model.sample(n)[0]
        else:
            raise NotImplementedError


class BackgroundModel(GenerativeModel):
    """
    Background/universal model P_u(·|c)
    """
    
    def __init__(
        self,
        model_type: str = "gmm",
        n_components: int = 10,
        **kwargs
    ):
        """
        Args:
            model_type: Type of model ("gmm", "flow", "nn")
            n_components: Number of components for GMM
        """
        self.model_type = model_type
        self.n_components = n_components
        self.models: Dict[str, Any] = {}  # Store per-context models
        self.kwargs = kwargs
    
    def fit(self, y_data: np.ndarray, c: Optional[str] = None):
        """Fit model for context c"""
        key = c if c else "default"
        
        if self.model_type == "gmm":
            model = GaussianMixture(
                n_components=self.n_components,
                **self.kwargs
            )
            model.fit(y_data)
            self.models[key] = model
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
    
    def log_prob(self, y: np.ndarray, z: Optional[str] = None, 
                c: Optional[str] = None) -> np.ndarray:
        """Compute log P_u(y | c)"""
        key = c if c else "default"
        
        if key not in self.models:
            raise ValueError(f"Model for context {c} not fitted")
        
        model = self.models[key]
        
        if self.model_type == "gmm":
            return model.score_samples(y.reshape(1, -1) if y.ndim == 1 else y)
        else:
            raise NotImplementedError
    
    def sample(self, n: int, z: Optional[str] = None, 
              c: Optional[str] = None) -> np.ndarray:
        """Sample from P_u(·|c)"""
        key = c if c else "default"
        
        if key not in self.models:
            raise ValueError(f"Model for context {c} not fitted")
        
        model = self.models[key]
        
        if self.model_type == "gmm":
            return model.sample(n)[0]
        else:
            raise NotImplementedError


