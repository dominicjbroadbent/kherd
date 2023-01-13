import numpy as np
from scipy.stats import multivariate_normal

def gmm_sample(probs, mus, covs, N):
    assert probs.shape[0] == mus.shape[0], 'The number of sampling probabilities must equal the number of means'
    assert probs.shape[0] == covs.shape[0], 'The number of sampling probabilities must equal the number of covariances'
    assert np.isclose(probs.sum(), 1), 'The sampling probabilities should sum to one' 
    assert isinstance(N, int) and N > 0, 'The number of samples must be a positive integer'
    
    X = np.zeros( ( 1, mus.shape[1]) )
    _, mixture_counts = np.unique(np.random.choice( probs.shape[0], N, p = probs), return_counts = True ) 
    for i in range( probs.shape[0] ):
        X = np.vstack( ( X, np.random.default_rng().multivariate_normal( mus[i], np.atleast_2d(covs[i]), size = mixture_counts[i] ) ) )
    
    # Remove beginning empty row and shuffle
    X = X[1:, :]
    np.random.shuffle(X)
    return X

def gmm_density(x, wts, mus, covs):
    assert wts.shape[0] == mus.shape[0], 'The number of GMM weights must equal the number of means'
    assert wts.shape[0] == covs.shape[0], 'The number of GMM weights must equal the number of covariances'
    assert np.isclose(wts.sum(), 1), 'The GMM weights should sum to one' 
    
    f = 0
    for i in range( wts.shape[0] ):
        f += wts[i] * multivariate_normal.pdf( x, mean = mus[i], cov = covs[i] )
    
    return f
