"""Provides common procrustes alignment related operations and algorithms.

For geometric morphometrics based studies, after landmark data are 
collected for each specimen, a typical next step is to remove the position, 
size and orientation information from the landmark set of each specimen so
that what remains is the shape information. This can be achieved by, for 
example, running Generalized Procrustes Aligment (see :func:`gpa()`) on the set 
of landmark sets.

After procrustes alignment, the shapes lie in a high-dimensional non-euclidean
manifold but are usually quite close to each other and can be projected to a
euclidean tangent space at their shape mean, whereupon they can be subjected to
multivariate analysis techniques like Principal Components Analysis, Partial 
Least Squares, etc.
"""

import numpy as np
import math
import morphops.lmk_util as lmk_util
import warnings

def get_position(lmks):    
    """Returns the centroid of the set or sets of landmarks in `lmks`.

    The centroid of a :math:`p` landmarks is simply the arithmetic mean of all 
    the landmark positions. That is 

    .. math:: \mathbf{x_c} = \sum_{i=1}^p \dfrac{\mathbf{x_i}}{p}

    Parameters
    ----------
    lmks : np.ndarray or list

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    centroid : np.array

        * If `lmks` is a (p,k) array, then `centroid` is a (k,)-shaped array,
          whose i-th element is the mean of the i-th coordinate in `lmks`.

        * If `lmks` is a (n,p,k) array, then `centroid` is a (n,k)-shaped
          array whose i-th element is the (k,)-shaped centroid of the i-th
          specimen's landmarks in `lmks`.
    """
    lmks_shape_dim = len(np.shape(lmks))
    if (lmks_shape_dim != 2) and (lmks_shape_dim !=3):
        raise ValueError("Input lmks must have either 2 size dimensions for a single specimen or 3 size dimensions for multiple specimens. Instead got {dims:d}.".format(dims=lmks_shape_dim))
    axis = lmks_shape_dim - 2
    return np.asarray(np.nanmean(lmks, axis=axis))

def get_scale(lmks):       
    """Returns the euclidean norm of the real matrix or matrices in `lmks`.

    The euclidean norm of the real (p x k) matrix :math:`X` is calculated as

    .. math:: \|X\| = \sqrt{Tr(X^T X)}

    Note
    ----
    `lmks` is not assumed to have been centered. To center the
    landmarks you can call `remove_position(lmks)`.

    Todo
    ----
    1. Check the literature to see if this is indeed meant to be the euclidean norm as opposed to the frobenius norm (I imagine it only differs if data is complex).

    Parameters
    ----------
    lmks : np.ndarray or list

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    scale : float or np.ndarray

        * If `lmks` is (p,k)-shaped, `scale` is a float representing its
          euclidean norm.
        
        * If `lmks` is (n,p,k)-shaped, `scale` is an (n,)-shaped array such 
          that the i-th element is the euclidean norm of the i-th specimen's
          landmarks.
    """
    lmks_shape_dim = len(np.shape(lmks))
    if (lmks_shape_dim != 2) and (lmks_shape_dim !=3):
        raise ValueError("Input lmks must have either 2 size dimensions for a single specimen or 3 size dimensions for multiple specimens. Instead got {dims:d}.".format(dims=lmks_shape_dim))
    axis = None if lmks_shape_dim == 2 else (1,2)
    return np.linalg.norm(lmks, axis=axis)

def remove_position(lmks, position=None):
    """If `position` is None, :func:`remove_position` translates `lmks` such 
    that :func:`get_position()` of `translated_lmks` is the origin. Else it is 
    the (:func:`get_position()` of `lmks`) - `position`.

    Parameters
    ----------
    lmks : np.ndarray or list

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    translated_lmks: np.array

        * **Single specimen** If `lmks` is (p,k)-shaped, `translated_lmks` is
          (p,k)-shaped such that the centroid of `translated_lmks` + `position`
          = centroid of `lmks`. When `position` is None, it is taken to be the
          centroid of `lmks`, which means `translated_lmks` is at the origin.
        
        * If `lmks` is (n,p,k)-shaped, `translated_lmks` is (n,p,k)-shaped such
          that the i-th element of `translated_lmks` is related to the i-th
          specimen of `lmks` by a translation calculated as per the single
          specimen case.
    """
    lmks_shape_dim = len(np.shape(lmks))
    pos = np.array(position) if position is not None else get_position(lmks)
    if lmks_shape_dim == 2:
        return lmks - pos
    else:
        return lmks - pos[:, np.newaxis, :]
    
def remove_scale(lmks, scale=None):
    """Scales the landmarks in `lmks` such that :func:`get_scale()` equals the euclidean norm divided by `scale` if `scale` is valid, else 1.

    Note `lmks` is not assumed to have been centered. To center the
    landmarks you may call `remove_position(lmks)`.

    Parameters
    ----------
    lmks : np.ndarray or list

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    np.array
        One of
        1. A (nl x d) landmark set with scale euclidean_norm/`scale` if `scale` is valid, else 1.
        2. A (k x nl x d) set of k landmark sets, each with scale (respective euclidean_norm/`scale`) if `scale` is valid, else 1.
    """
    scale_ = scale if (scale is not None) else get_scale(lmks)
    lmks_shape = np.shape(lmks)
    num_lmksets = 1 if len(lmks_shape) == 2 else lmks_shape[0]
    scale_re = np.reshape(scale_, (num_lmksets,1,1))
    return np.reshape(np.divide(lmks, scale_re), lmks_shape)

def rotate(source, target, no_reflect=False):
    """Rotates the landmark set `source` so as to minimize its sum of squared interlandmark distances to `target`.

    By default `rotate` will also reflect `source` if it offers a better alignment to `target`. This behavior can be switched off by setting the optional parameter `no_reflect` to True, in which case `source` will be aligned to `target` using a pure rotation belonging in SO(d).

    ### Todo
    1. Handle when values are NaN.

    ### References
    1. Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." no 3 (2017): 1-5. 
    I found a pdf here: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    Parameters
    ----------
    source : np.ndarray or list
        A (nl x d) set of landmarks corresponding to the source shape.

    target : np.ndarray or list
        A (nl x d) set of landmarks corresponding to the target shape.

    no_reflect : bool, optional
        Flag indicating whether the best alignment should exclude reflection (default is False, which means reflection will be used if it achieves better alignment).

    Returns
    -------
    result: dict
        src_ald: np.array
            A (nl x d) landmark set consisting of the `source` landmarks rotated to the `target`.
        R: np.array
            A (d x d) array representing the right rotation matrix.
        D: np.array
            A (1 x d) array representing the diagonal matrix of the SVD of np.dot(target.T, source).
    """
    result = {'src_ald': None, 'R': None, 'D': None}
    # Get the (d x d) covariance between target and source.
    C = np.matmul(lmk_util.transpose(target), source)
    # Need argmax of tr(Y(XR)t) = tr(RYtX) = tr(RC). Let svd(C) = UDVt.
    U, D, VT = np.linalg.svd(C)
    V = lmk_util.transpose(VT)
    UT = lmk_util.transpose(U)
    # Then tr(RC) = tr(R(UDVt)) = tr(D(VtRU)). But M=VtRU is orthogonal and D is non-negative diagonal, so argmax occurs when M = I => R = VUt. We done?
    R = np.matmul(V, UT)
    # Well, the above R is not guaranteed to be in SO(d), only in O(d). det(R) is 1 when R is a rotation, else -1 when R is a reflection.
    detR = np.linalg.det(R)
    # Say det(R) = det(VUt) = -1. If we want to force det(R) = 1, then det(M) = det(VtRU) = det(R)*det(VUt) = -1.
    # So if R is constrained to be a rotation, M must be a reflection.
    ndet_i = detR < 0
    if no_reflect and np.any(ndet_i):
        # Notice argmax tr(DM) = sum d_ii*m_ii is a convex function f(m00, m11, ..) on the set of diagonals of reflection matrices. This set is the convex hull of E = (+-1,+-1,..), where the num of -1s is odd per A.Horn (1954). So f is maximized at a vertex in E, but where? 
        ones = np.ones(lmk_util.num_coords(source))
        # Well every dii is non-negative, so assuming that dii are in descending order: argmax f has to be (1,1,..,1,-1).
        ones[-1] = -1
        # Say N = diag(1,1,..,-1).
        N = np.diag(ones)
        # Then M = VtRU = N => R = VNUt.
        R[ndet_i] = np.matmul(np.matmul(V[ndet_i], N), UT[ndet_i])
        # Also update D
        D[ndet_i] = np.multiply(D[ndet_i], ones)
    result['src_ald'] = np.matmul(source, R)
    result['R'] = R
    result['D'] = D
    return result

def opa(source, target, do_scaling=False, no_reflect=False):
    """Perform ordinary procrustes alignment from source to target.

    ...todo...::
        Handle degenerate source, target landmarks.
        Handle fewer landmarks in source.
    """
    result = { 'oss': None, 'oss_stdized': None, 'b': None, 'R': None, 'c': None, 'src_ald': None }
    # 1. Remove position information
    muX = get_position(source)
    X0 = remove_position(source, muX)
    muY = get_position(target)
    Y0 = remove_position(target, muY)
    # 2. Remove scale information
    X0_norm = get_scale(X0)
    Y0_norm = get_scale(Y0)
    X0 = remove_scale(X0, X0_norm)
    Y0 = remove_scale(Y0, Y0_norm)
    # 2.i. Also keep the squared norm.
    X0_ssq = np.square(X0_norm)
    Y0_ssq = np.square(Y0_norm)
    # 3. Rotate source to target
    rot_res = rotate(X0, Y0, no_reflect=no_reflect)
    result['R'] = rot_res['R']
    # For all further comments here, assume X, Y are centered.
    # 4. For scaling and OSS calculation, we note that 
    # D^2_opa(X, Y) = ||Y - bXR - 1*c.T||^2 = tr(||Y||^2 + b^2||X||^2 - 2b*Y.T*X*R) + d*c.T*c.
    # Say X0, Y0 are preshapes (as is true here).
    # D^2 = tr(||Y||^2 + b^2||X||^2 - 2b*||Y||*||X||*Y0.T*X0*R) + d*c.T*c.
    # Diff wrt b gives dD^2/db = 2b*tr(X.T*X) - 2*tr(||Y||*||X||*Y0.T*X0*R)
    # So b = ||Y||*tr(Y0.T*X0*R)/||X|| = ||Y||*tr(U*D*V.T*V*U.T)/||X|| = ||Y||*tr(D)/||X||.
    traceD = np.sum(rot_res['D'])
    if do_scaling:
        result['b'] = (Y0_norm*traceD)/X0_norm
        # Also, cos(rho(X,Y)) = tr(D), and oss = ||Y^2||sin^2(rho(X,Y))
        # So oss = ||Y^2||(1-cos^2(rho(X,Y))) = ||Y^2||(1-tr(D)^2)
        # For standardized oss we divide by ||Y^2||.
        result['oss_stdized'] = 1 - (traceD*traceD)
        result['oss'] = Y0_ssq*result['oss_stdized']
        result['src_ald'] = \
            remove_position(np.dot(Y0_norm*traceD*X0,result['R']), -muY)
    else:
        result['b'] = 1
        # The oss expression with a given b is 
        # ||Y||^2 + 2*b^2*||X||^2 - 2*b*||X||*||Y||*cos(rho(X,Y))
        # Again for standardized oss we divide by ||Y^2||.
        result['oss_stdized'] = 1 + (X0_ssq/Y0_ssq) - (2*(X0_norm/Y0_norm)*traceD)
        result['oss'] = Y0_ssq*result['oss_stdized']
        result['src_ald'] = \
            remove_position(np.dot(X0_norm*X0,result['R']), -muY)
    # c is the gap between centroids of bXR and Y.
    result['c'] = muY - result['b']*np.dot(muX, result['R'])
    
    return result

def gpa(X, tol=1e-5,max_iters=10, do_project=False, do_scaling=False,
        no_reflect=False, unitize_mean=False):
    """Perform Generalized Procrustes Alignment on all lmk sets in X.
    """
    res = {'X0_ald': None, 'X0_ald_mu': None, 'X0_b': None, 'ssq': None}
    n_lmk_sets = lmk_util.num_lmk_sets(X)
    n_lmks = lmk_util.num_lmks(X)
    n_coords = lmk_util.num_coords(X)

    # 1. Remove position
    muX = get_position(X)
    X0 = remove_position(X, muX)
    
    # 2. Remove scale (if not do_scaling, we're just doing partial procrustes)
    X0_norm = get_scale(X0)
    X0 = remove_scale(X0, X0_norm)
    X0_b = np.reciprocal(X0_norm)
    
    # 3. Rotate all lmk sets to the mean of all other lmk sets. Scale.
    X0_ald = X0
    ssq, ssq_old = None, None
    curr_iter = 0
    all_i = np.arange(n_lmk_sets)

    def is_ssq_ok():
        return ((ssq is not None) and (ssq_old is not None) and 
               ((ssq_old - ssq) >= 0) and
               ((ssq_old - ssq) <= tol))
    
    while (not is_ssq_ok()) and (curr_iter < max_iters):
        # 3.1. Rotate
        while(not is_ssq_ok()):
            ssq_old = ssq
            for i in range(n_lmk_sets):
                # Get the mean of all but the ith lmk set
                all_but_i = X0_ald[all_i != i]
                mean_for_i = (1.0/(n_lmk_sets-1))*np.sum(all_but_i, axis=0)
                # Rotate all lmk sets to this mean
                X0_ald = rotate(X0_ald, mean_for_i, no_reflect)['src_ald']
            ssq = get_ssqd(X0_ald)

        # 3.2. Scale
        if do_scaling:
            # We first get the biggest eigvec the nxn corr matrix.
            X0_ald_vecd = np.reshape(X0_ald, (n_lmk_sets, n_coords*n_lmks))
            X0_corrcoef = np.corrcoef(X0_ald_vecd)
            eig_vals, eig_vecs = np.linalg.eigh(X0_corrcoef)
            sort_perm = eig_vals.argsort()
            phi = eig_vecs[:, sort_perm][:, -1]
            if np.all(phi < 0):
                phi = np.abs(phi) # TODO: Is this okay to do?
            # The scale beta_i = sqrt(sum of sqd norms/ith sqd norm)*phi[i]
            X0_ald_norm = get_scale(X0_ald)
            X0_ald_ssq_norm = np.sqrt(np.sum(np.square(X0_ald)))
            frac = np.reciprocal(X0_ald_norm, dtype=np.float64)*X0_ald_ssq_norm
            beta = np.multiply(frac, phi)
            # Rescale X0_ald[i] by b_i
            X0_ald = np.multiply(X0_ald, np.reshape(beta, (n_lmk_sets,1,1)))
            # Update X0_b
            X0_b = np.multiply(X0_b, beta)

        ssq = get_ssqd(X0_ald)
        curr_iter += 1
    
    print("ssq diff", ssq_old - ssq)

    # The mean is just the mean of the procrustes aligned lmk sets.
    X0_ald_mu = (1.0/n_lmk_sets)*np.sum(X0_ald, axis=0)
    if unitize_mean:
        X0_ald_mu = remove_scale(X0_ald_mu)

    if do_project:
        if do_scaling:
            w_msg = ("`do_project` assumes that the aligned lmk sets are scaled to have unit centroid size, which is not guaranteed if `do_scaling`. Proceeding with projection using the non-unit size lmk sets. See \'Rohlf, F. J. (1999). Shape statistics: Procrustes superimpositions and tangent spaces.\'")
            warnings.warn(w_msg)
        XC = X0_ald_mu.reshape((1, n_coords*n_lmks))
        X = X0_ald.reshape((n_lmk_sets, n_coords*n_lmks))
        # Get the projection matrix to project a shape onto X_c.
        XC_proj = (1.0/(XC @ XC.T)) * (XC.T @ XC)
        # Project all shapes onto the subspace orthogonal to X_c.
        X_tan = X @ (np.identity(n_coords*n_lmks) - XC_proj)
        # The above are like coordinates in the tangent space.
        # To get the "icons", we add back the mean.
        X0_ald = (X_tan + XC).reshape((n_lmk_sets, n_lmks, n_coords))
        # Recalculate the ssq
        ssq = get_ssqd(X0_ald)
        print("ssq diff", ssq_old - ssq)

    res['X0_ald'] = X0_ald
    res['X0_ald_mu'] = X0_ald_mu
    res['X0_b'] = X0_b
    res['ssq'] = ssq
    return res

def get_ssqd(X):
    """Alias for `lmk_util.ssqd(X)`.
    """
    return lmk_util.ssqd(X)

