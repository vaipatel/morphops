"""Provides thin-plate splines related operations and algorithms.

Given two sets of points, the thin-plate spline can interpolate from one to the 
other in a manner that minimizes the "integral bending norm"[bookstein89]_.

Importantly, it has a remarkable connection to Kendall's shape space in the 
following way: The non-zero eigenvectors of the bending energy matrix form an 
orthonormal basis in the tangent space of shape coordinates [bookstein96]_.

References
----------
.. [bookstein89] Bookstein, F.L., 1989. Principal warps: Thin-plate splines 
    and the decomposition of deformations. IEEE Transactions on pattern 
    analysis and machine intelligence, 11(6), pp.567-585.
.. [bookstein96] Bookstein, F.L., 1996. Biometrics, biomathematics and the 
    morphometric synthesis. Bulletin of mathematical biology, 58(2), p.313.
"""

import numpy as np
import math
import morphops.lmk_util as lmk_util
import warnings

def K_matrix(X, Y=None):
    """Calculates the upper-right (p,p) submatrix of the (p+k+1,p+k+1)-shaped 
    L matrix.

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like

        A (p,k) array of p points in k=2 or k=3 dimensions.

    Y : (m,2) or (m,3) shaped array-like, optional
        A (m,k) array of p points in k=2 or k=3 dimensions. `Y` must have the 
        same k as `X`.

        If `Y` is `None`, it is just set to `X`.
    
    Returns
    -------
    K : np.ndarray
        A (p,p) array where the element at [i,j] is :math:`U(\|X_i - Y_j\|)`. The definition of U depends on k. 
        
        In particular, if k = 2, then  :math:`U(r) = r^2 \log(r^2)`, else 
        :math:`U(r) = r`.

        Note: Using :math:`\\alpha U(r)` instead of :math:`U(r)` for some 
        :math:`\\alpha \in \mathbb{R}` will not change the calculated spline.
        Simple block matrix inverse formulae show that when calculating :math:`L^{-1}`
        for the spline using :math:`\\alpha U(r)`, the non-uniform coefficients
        multiplied to the :math:`U` terms will be scaled by :math:`\\frac{1}{\\alpha}`
        while the uniform coefficients will stay the same.
    """
    num_coords = lmk_util.num_coords(X)
    if (num_coords != 2) and (num_coords != 3):
        raise ValueError("The input matrix must have landmarks with "
                         "coordinates in either 2 or 3 dimensions.")
    if Y is None:
        Y = X
    r = lmk_util.distance_matrix(X, Y)
    if (num_coords == 2):
        r_sqd = np.square(r)
        # Make a copy of r_sqd where 0->1. This copy will be passed to log.
        # This way log(1) will be 0 and we wont get NaN and warnings. 
        r_sqd_cl = np.copy(r_sqd)
        r_sqd_cl[np.isclose(r_sqd_cl,0)] = 1
        return np.multiply(r_sqd, np.log(r_sqd_cl))
    # else num_coords is 3
    return r

def P_matrix(X):
    """Makes the minor diagonal submatrix P of the (p+k+1,p+k+1)-shaped L 
    matrix.

    Basically just stacks a column of 1s before the coordinate columns in `X`.

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like
        A (p,k) array of p points in k=2 or k=3 dimensions.

    Returns
    -------
    P : np.ndarray
        A (p,k+1) array, which is 1 in the first column, and exactly `X` in the 
        remaining columns.
    """
    ones = np.ones(lmk_util.num_lmks(X))
    return np.column_stack((ones, X))

def L_matrix(X):
    """Makes the (p+k+1,p+k+1)-shaped L matrix that gets inverted when 
    calculating the thin-plate spline "from" `X`.

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like
        A (p,k) array of p landmarks in k=2 or k=3 dimensions for one specimen.

    Returns
    -------
    L : np.ndarray
        A (p+k+1,p+k+1) array of the form [[K | P][P.T | 0]].
    """
    n_coords = lmk_util.num_coords(X)
    n_lmks = lmk_util.num_lmks(X)
    K = K_matrix(X)
    P = P_matrix(X)
    L = np.zeros((n_lmks + n_coords + 1, n_lmks + n_coords + 1))
    L[0:n_lmks,0:n_lmks] = K
    L[0:n_lmks,n_lmks:] = P
    L[n_lmks:,0:n_lmks] = np.transpose(P)
    return L

def bending_energy_matrix(X):
    """Returns the upper right (pxp) submatrix of L^(-1).

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like
        A (p,k) array of p landmarks in k=2 or k=3 dimensions for one specimen.

    Returns
    -------
    L_inv : np.ndarray
        The upper right (p,p) submatrix of the inverse of the `L_matrix` of `X`.
    """
    n_lmks = lmk_util.num_lmks(X)
    L = L_matrix(X)
    L_inv = np.linalg.inv(L)
    return L_inv[0:n_lmks,0:n_lmks]

def tps_coefs(X, Y):
    """Finds the thin-plate spline coefficients for the thin-plate spline 
    function that interpolates from X to Y.

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like
        A (p,k) array of p points in k=2 or k=3 dimensions.

    Y : (p,2) or (p,3) shaped array-like
        A (p,k) array of p points in k=2 or k=3 dimensions. `Y` must have the 
        same shape as `X`.

    Returns
    -------
    W : np.ndarray
        A (p,k) array of weights for the non-affine part of the spline.
    
    A : np.ndarray
        A (k+1,k) array of weights for the affine part of the spline.
    """
    n_coords = lmk_util.num_coords(X)
    n_lmks = lmk_util.num_lmks(X)
    Y_0 = np.row_stack((Y, np.zeros((n_coords+1,n_coords))))
    L = L_matrix(X)
    Q = np.linalg.solve(L, Y_0)
    if np.any(np.isnan(Q)):
        raise ValueError("The result of L_inv*Y contained NaN values.")
    # return W and A.
    return Q[0:n_lmks], Q[n_lmks:]

def tps_warp(X, Y, pts):
    """Maps points `pts` to their image under the thin-plate spline function generated by :func:`tps_coefs` of `X` and `Y`.

    Parameters
    ----------
    X : (p,2) or (p,3) shaped array-like
        A (p,k) array of p points in k=2 or k=3 dimensions.

    Y : (p,2) or (p,3) shaped array-like
        A (p,k) array of p points in k=2 or k=3 dimensions. `Y` must have the 
        same shape as `X`.

    pts : (m,2) or (m,3) shaped array-like, optional
        A (m,k) array of m points in k=2 or k=3 dimensions. `pts` must have the 
        same coordinate dimensions k as `X`.

    Returns
    -------
    warped_pts : (m,2) or (m,3) shaped array-like, optional
        A (m,k) array of points corresponding to the image of `pts` under the thin-plate spline produced by `X`, `Y`.
    """
    W, A = tps_coefs(X, Y)
    U = K_matrix(pts, X)
    P = P_matrix(pts)
    # The warped pts are the affine part + the non-uniform part
    return np.matmul(P,A) + np.matmul(U,W)
    

