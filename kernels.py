import numpy as np
from scipy.spatial.distance import cdist

class BaseKernel:
    def __init__(self):
        """Initialise the kernel instance"""
        pass
    
    def __call__(self, x, y):
        """Evaluate the kernel function at (x, y)"""
        raise NotImplementedError()
        
class Gaussian(BaseKernel):
    def __init__(self, lengthscale = 0.5):
        """Initialise the Gaussian kernel"""
        self.lengthscale = lengthscale
        super().__init__()
        
    def __call__(self, x, y):
        """Evaluate the kernel function at (x, y)"""
        sq_diffs = cdist(x, y, metric = 'sqeuclidean')
        return np.exp( - 0.5 * sq_diffs / self.lengthscale)
