import numpy as np

from tqdm.notebook import tqdm
from scipy.optimize import minimize 

from kernels import BaseKernel

class KernelHerding:
    def __init__(self, initial_samples, kernel):
        """Initialise the kernel herding class with samples from the distribution and the kernel"""
        
        assert initial_samples.ndim == 2, 'The array of initial samples must be two dimensional'
        assert initial_samples.shape[0] > 0, 'You must provide a nonzero number of samples from the distribution of interest'
        self.initial_samples = initial_samples # N x p
        
        # Empty arrays to store super samples (and idcs if we optimise locally)
        self.super_samples = np.empty( (0, initial_samples.shape[1] ) )
        self.super_sample_idcs = np.array([], dtype = np.int32)
        
        assert isinstance(kernel, BaseKernel), 'The kernel must be an instance of BaseKernel'
        self.kernel = kernel
    
    def reset(self):
        """Reset the super samples and any corresponding local idcs"""
        self.super_samples = np.empty( (0, self.initial_samples.shape[1] ) )
        try:
            del self._super_sample_sums 
        except AttributeError:
            pass
        self.super_sample_idcs = np.array([], dtype = np.int32)
    
    def set_initial_samples(self, initial_samples):
        """Set initial_samples"""
        assert initial_samples.ndim == 2, 'The array of initial samples must be two dimensional'
        assert initial_samples.shape[0] > 0, 'You must provide a nonzero number of samples from the distribution of interest'
        self.initial_samples = initial_samples # N x p
        self.delete_stored_expectations()
    
    def set_kernel(self, kernel):
        """Set kernel function"""
        assert isinstance(kernel, BaseKernel), 'The kernel must be an instance of BaseKernel'
        self.kernel = kernel
        self._delete_stored_expectations()
    
    def _delete_stored_expectations(self):
        """Delete stored expectations if they exist"""
        try:
            del self.expectations 
        except AttributeError:
            pass

    def herd(self, itrs, method):
        """
        Do kernel herding for a specified number of iterations, optimising over the entire space globally, or 
        restricting attention locally to just the initial samples.
        
        For local herding then the method tuple should have the form
                    method  = ('local', True) or ('local', False)
        where the boolean corresponds to restricting the super-samples to be unique (True) or not (False)
        
        For global herding then the method tuple should have the form
                    method  = ('global', 'opt_method', num_samples)
        where the 'opt_method' string corresponds to a scipy.optimize.minimize optimisation method and num_samples
        is the number of initial samples that should be used to find the best initial guess for each herding iteration.
        """
        assert isinstance(itrs, int) and itrs > 0, 'The number of iterations must be a positive integer'
        
        if method[0] == 'global':
            for i in tqdm(range(itrs)):
                self.super_samples = np.append( self.super_samples, self._globally_herd_next_sample(method), axis = 0 )
            
        elif method[0] == 'local':
            for i in tqdm(range(itrs)):
                # Get the next super sample and its index from the initial samples
                super_sample, super_sample_index = self._locally_herd_next_sample(method[1])
                self.super_samples = np.append( self.super_samples, super_sample, axis = 0 ) 
                self.super_sample_idcs = np.append( self.super_sample_idcs, super_sample_index)
        
    def _globally_herd_next_sample(self, method):
        """Choose the next best super sample by globally optimising the kernel herding objective function"""
        def objective(x):
            x = np.atleast_2d(x)
            value = self.kernel(x, self.initial_samples).mean(axis = 1)
            value -= ( 1 / ( self.super_samples.shape[0] + 1 ) ) * self.kernel(x, self.super_samples).sum(axis = 1)
            return -value
            
        x0 = self._get_best_initial_guess(self.initial_samples, method[2], objective)
        res = minimize(objective, x0, method = method[1])
            
        if res['message'] != 'Optimization terminated successfully.':
            raise Exception(f'Optimisation error: {res["message"]}')
            
        return np.atleast_2d(res['x'])
            
    def _get_best_initial_guess(self, samples, num_samples, objective):
        # Get num_samples points from the initial samples randomly
        idcs =  np.random.choice(samples.shape[0], num_samples, replace = False)
        pts = samples[idcs, :]
            
        return pts[ np.argmin( objective( pts ) ), : ]
    
    def _locally_herd_next_sample(self, unique):
        """Choose the next best super sample from the set of initial samples"""
        num_samples = self.initial_samples.shape[0]
        
        # Approximate the expectation of the kernel function evaluated at each initial sample, do this only once as these stay the same
        try:
            self.expectations 
        except AttributeError:
            print('Computing and storing local expectations...')
            # self.expectations = self._kernel_expectation( self.initial_samples )
            self.expectations = np.zeros(num_samples)
            for i in tqdm(range(num_samples)):
                self.expectations[i] = self.kernel(self.initial_samples[[i], :], self.initial_samples).mean()                
            print('Done! \nHerding...')
        
        # Keep track of the sum of the kernel evaluations of each initial sample with each of the super samples so 
        # we don't have to keep recomputing and resumming over previous super samples
        try:
            self._super_sample_sums 
        except AttributeError:
            self._super_sample_sums = np.zeros(num_samples)
            
        objective = np.zeros(num_samples)    
        for i in range(num_samples):
            # Add the kernel evaluation at the ith initial sample and the most recent super-sample
            if self.super_samples.shape[0] > 0:
                self._super_sample_sums[i] += self.kernel( self.initial_samples[[i], :], self.super_samples[[-1], :] )
            # Compute the objective function value for each initial sample
            objective[i] =  self.expectations[i] - ( ( 1 / ( self.super_samples.shape[0] + 1 ) ) * self._super_sample_sums[i] )
            
        # Make sure the super-samples are unique (if wanted)
        if unique == True:
            objective[self.super_sample_idcs] = -np.inf
            
        argmax_index = np.argmax(objective)
        super_sample = self.initial_samples[[argmax_index], :]
        super_sample_index = argmax_index
        
        return super_sample, super_sample_index    
    
    def compute_herd_error(self, f):
        """Compute the RMSE wrt a function f of super samples versus empirical distribution"""
        mu_p = f( np.mean(self.initial_samples, axis = 0) )
        
        herd_error = np.zeros( ( self.super_samples.shape[0], self.super_samples.shape[1] ) )
        super_sums = np.cumsum( f( self.super_samples ), axis = 0)

        for d in range(self.super_samples.shape[1]):
            for i in range(self.super_samples.shape[0]):
                herd_error[i, d] = ( mu_p[d] - ( super_sums[i, d] / ( i + 1 ) ) )**2
        
        return np.sqrt( ( 1 / self.super_samples.shape[1] ) * np.sum( herd_error, axis = 1 ) )
        
    def compute_iid_error(self, f):
        """Compute the RMSE wrt a function f of iid samples versus empirical distribution"""
        # Compute the sample mean   
        mu_p = f( np.mean(self.initial_samples, axis = 0) )
        
        np.random.shuffle(self.initial_samples)
        iid_error = np.zeros( ( self.initial_samples.shape[0], self.initial_samples.shape[1] ) )
        iid_sums =  np.cumsum( f( self.initial_samples ), axis = 0)

        for d in range(self.initial_samples.shape[1]):
            for i in range(self.super_samples.shape[0]):
                iid_error[i, d] = ( mu_p[d] - ( iid_sums[i, d] / ( i + 1 ) ) )**2
        
        return np.sqrt( ( 1 / self.initial_samples.shape[1] ) * np.sum( iid_error, axis = 1 ) )
         
