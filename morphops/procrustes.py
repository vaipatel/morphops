import numpy as np
import math

def get_position(lmks):    
    """Returns the centroid of the set or sets of landmarks in `lmks`.

    Parameters
    ----------
    lmks : np.ndarray or list
        One of
        1. A (nl x d) set of landmarks corresponding to a single specimen.
        2. A (k x nl x d) set of k landmark sets corresponding to k specimens.

    Returns
    -------
    centroid : np.array
        One of
        1. A d-dimensional array whose ith entry is the mean of the ith landmark coordinate in `lmks`.
        2. A (k x d)-dimensional array whose kth element is the d-dimensional centroid of the kth specimen's landmarks in `lmks`.
    """
    lmks_shape_dim = len(np.shape(lmks))
    if (lmks_shape_dim != 2) and (lmks_shape_dim !=3):
        raise ValueError("Input lmks must have either 2 size dimensions for a single specimen or 3 size dimensions for multiple specimens. Instead got {dims:d}.".format(dims=lmks_shape_dim))
    axis = lmks_shape_dim - 2
    return np.asarray(np.nanmean(lmks, axis=axis))

def get_scale(lmks):       
    """Returns the euclidean norm of the matrix or matrices in `lmks`.

    Note `lmks` is not assumed to have been centered. To center the
    landmarks you may call `remove_position(lmks)`.

    ### Todo
    1. Check the literature to see if this is indeed meant to be the euclidean norm as opposed to the frobenius norm (I imagine it only differs if data is complex).

    Parameters
    ----------
    lmks : np.ndarray or list
        One of
        1. A (nl x d) set of landmarks corresponding to a single specimen.
        2. A (k x nl x d) set of k landmark sets corresponding to k specimens.

    Returns
    -------
    scale : np.float64 or np.ndarray
        One of
        1. A single float for the euclidean norm of the `lmks` matrix.
        2. A k-dimensional array whose kth element is the frobenius norm of the kth specimen's landmarks matrix in the `lmks` matrices.
    """
    lmks_shape_dim = len(np.shape(lmks))
    if (lmks_shape_dim != 2) and (lmks_shape_dim !=3):
        raise ValueError("Input lmks must have either 2 size dimensions for a single specimen or 3 size dimensions for multiple specimens. Instead got {dims:d}.".format(dims=lmks_shape_dim))
    axis = None if lmks_shape_dim == 2 else (1,2)
    return np.linalg.norm(lmks, axis=axis)

def remove_position(lmks, position=None):
    """Translates the landmarks in `lmks` such that `get_position()` coincides with (centroid - `position`) if `position` is valid, else the origin.

    ...todo...:: Do a better check of position.

    Parameters
    ----------
    lmks : np.ndarray or list
        One of
        1. A (nl x d) set of landmarks corresponding to a single specimen.
        2. A (k x nl x d) set of k landmark sets corresponding to k specimens.

    Returns
    -------
    np.array
        One of
        1. A (nl x d) landmark set centered on (centroid - `position`) if `position` is valid, else the origin.
        2. A (k x nl x d) set of k landmark sets, each centered on their (respective centroid - `position`) if `position` is valid else the origin.
    """
    lmks_shape_dim = len(np.shape(lmks))
    pos = np.array(position) if position is not None else get_position(lmks)
    if lmks_shape_dim == 2:
        return lmks - pos
    else:
        return lmks - pos[:, np.newaxis, :]
    
def remove_scale(lmks, scale=None):
    """Scales the landmarks in `lmks` such that `get_scale()` equals the euclidean norm divided by `scale` if `scale` is valid, else 1.

    Note `lmks` is not assumed to have been centered. To center the
    landmarks you may call `remove_position(lmks)`.

    Parameters
    ----------
    lmks : np.ndarray or list
        One of
        1. A (nl x d) set of landmarks corresponding to a single specimen.
        2. A (k x nl x d) set of k landmark sets corresponding to k specimens.

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
    C = np.dot(target.T, source)
    # Need argmax of tr(Y(XR)t) = tr(RYtX) = tr(RC). Let svd(C) = UDVt.
    U, D, VT = np.linalg.svd(C)
    # Then tr(RC) = tr(R(UDVt)) = tr(D(VtRU)). But M=VtRU is orthogonal and D is non-negative diagonal, so argmax occurs when M = I => R = VUt. We done?
    R = np.dot(VT.T, U.T)
    # Well, the above R is not guaranteed to be in SO(d), only in O(d). det(R) is 1 when R is a rotation, else -1 when R is a reflection.
    detR = np.linalg.det(R)
    # Say det(R) = det(VUt) = -1. If we want to force det(R) = 1, then det(M) = det(VtRU) = det(R)*det(VUt) = -1.
    # So if R is constrained to be a rotation, M must be a reflection.
    if no_reflect and detR < 0:
        # Notice argmax tr(DM) = sum d_ii*m_ii is a convex function f(m00, m11, ..) on the set of diagonals of reflection matrices. This set is the convex hull of E = (+-1,+-1,..), where the num of -1s is odd per A.Horn (1954). So f is maximized at a vertex in E, but where? 
        ones = np.ones(VT.shape[0])
        # Well every dii is non-negative, so assuming that dii are in descending order: argmax f has to be (1,1,..,1,-1).
        ones[-1] = -1
        # Say N = diag(1,1,..,-1).
        N = np.diag(ones)
        # Then M = VtRU = N => R = VNUt.
        R = np.dot(np.dot(VT.T, N), U.T)
        # Also update D
        D[-1] *= -1
    result['src_ald'] = np.dot(source, R)
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
