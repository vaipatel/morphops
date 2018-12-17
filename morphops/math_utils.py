import numpy as np

def transpose(X):
    """Swaps the last two axes of a N-D tensor.

    So for a 2-D matrix, this returns the transpose.
    For a 3-D tensor of length `n`, this returns the array of `n` 
    transposed matrices.
    """
    X_d_sz = len(np.shape(X))
    if (len(X) < 2):
        return X
    return np.swapaxes(X, X_d_sz - 2, X_d_sz - 1)