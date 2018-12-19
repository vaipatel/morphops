import numpy as np

def num_lmk_sets(X):
    if (len(np.shape(X)) is not 3):
        raise ValueError("The input X must be a 3d tensor corresponding to a "
                         "list of landmark sets.")
    return np.shape(X)[0]

def num_lmks(X):
    X_shape = np.shape(X)
    X_d_sz = len(X_shape)
    return X_shape[X_d_sz - 2]

def num_coords(X):
    X_shape = np.shape(X)
    X_d_sz = len(X_shape)
    return X_shape[X_d_sz - 1]

def transpose(X):
    """Swaps the last two axes of a N-D tensor.

    So for a 2-D matrix, this returns the transpose.
    For a 3-D tensor of length `n`, this returns the array of `n` 
    transposed matrices.
    """
    X_d_sz = len(np.shape(X))
    if (X_d_sz < 2):
        return X
    return np.swapaxes(X, X_d_sz - 2, X_d_sz - 1)

def ssqd(X):
    """Returns the average sum of squared norms of pairwise differences 
       between all lmk sets in X.
    """
    n_lmk_sets = num_lmk_sets(X)
    if (n_lmk_sets < 2):
        raise ValueError("The input X must contain atleast 2 landmark sets.")
    ssq = 0
    for i in np.arange(n_lmk_sets - 1):
        ssq += np.sum(np.square(X[i:] - X[i]))
    return ssq*1.0/n_lmk_sets