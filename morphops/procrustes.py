"""Provides procrustes alignment related operations and algorithms.

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
    lmks : array-like

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    centroid : numpy.ndarray

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
    `lmks` is not assumed to have been pre-centered. To pre-center `lmks` you 
    can call :func:`remove_position` on `lmks` before applying `remove_scale`.

    Todo
    ----
    1. Check the literature to see if this is indeed meant to be the euclidean norm as opposed to the frobenius norm (I imagine it only differs if data is complex).

    Parameters
    ----------
    lmks : array-like

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    scale : numpy.float64 or numpy.ndarray

        * **Single specimen** If `lmks` is (p,k)-shaped, `scale` is a float
          representing its euclidean norm.
        
        * **n specimens** If `lmks` is (n,p,k)-shaped, `scale` is an (n,)
          -shaped array such that the i-th element is the euclidean norm of the
          i-th specimen's landmarks.
    """
    lmks_shape_dim = len(np.shape(lmks))
    if (lmks_shape_dim != 2) and (lmks_shape_dim !=3):
        raise ValueError("Input lmks must have either 2 size dimensions for a single specimen or 3 size dimensions for multiple specimens. Instead got {dims:d}.".format(dims=lmks_shape_dim))
    axis = None if lmks_shape_dim == 2 else (1,2)
    return np.linalg.norm(lmks, axis=axis)

def remove_position(lmks, position=None):
    """If `position` is `None`, :func:`remove_position` translates `lmks` such 
    that :func:`get_position()` of `translated_lmks` is the origin. Else it is 
    the (:func:`get_position()` of `lmks`) - `position`.

    Parameters
    ----------
    lmks : array-like

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    translated_lmks: numpy.ndarray

        * **Single specimen** If `lmks` is (p,k)-shaped, `translated_lmks` is
          (p,k)-shaped such that the centroid of `translated_lmks` + `position`
          = centroid of `lmks`. When `position` is None, it is taken to be the
          centroid of `lmks`, which means `translated_lmks` is at the origin.
        
        * **n specimens** If `lmks` is (n,p,k)-shaped, `translated_lmks` is
          (n,p,k)-shaped such that the i-th element of `translated_lmks` is
          related to the i-th specimen of `lmks` by a translation calculated as
          per the single specimen case.
    """
    lmks_shape_dim = len(np.shape(lmks))
    pos = np.array(position) if position is not None else get_position(lmks)
    if lmks_shape_dim == 2:
        return lmks - pos
    else:
        return lmks - pos[:, np.newaxis, :]
    
def remove_scale(lmks, scale=None):
    """If `scale` is `None`, :func:`remove_scale` scales `lmks` such that 
    :func:`get_scale()` of `scaled_lmks` is 1. Else it is (:func:`get_scale` of 
    `lmks`)/`scale`.

    Note
    ----
    `lmks` is not assumed to have been pre-centered. To pre-center `lmks` you 
    can call :func:`remove_position` on `lmks` before applying `remove_scale`.

    Parameters
    ----------
    lmks : array-like

        One of the following
        
        * **Single specimen** A (p,k) array of p landmarks in k dimensions for
          one specimen.

        * **n specimens** A (n,p,k) array of n landmark sets for n specimens, 
          each having p landmarks in k dimensions.

    Returns
    -------
    scaled_lmks: numpy.ndarray

        * **Single specimen** If `lmks` is (p,k)-shaped, `scaled_lmks` is
          (p,k)-shaped such that the norm of `scaled_lmks` x `scale`
          = norm of `lmks`. When `scale` is `None`, it is taken to be the
          norm of `lmks`, which means `scaled_lmks` has norm 1.
        
        * **n specimens** If `lmks` is (n,p,k)-shaped, `scaled_lmks` is
          (n,p,k)-shaped such that the i-th element of `scaled_lmks` is
          related to the i-th specimen of `lmks` by a scaling calculated as
          per the single specimen case.
    """
    scale_ = scale if (scale is not None) else get_scale(lmks)
    lmks_shape = np.shape(lmks)
    num_lmksets = 1 if len(lmks_shape) == 2 else lmks_shape[0]
    scale_re = np.reshape(scale_, (num_lmksets,1,1))
    return np.reshape(np.divide(lmks, scale_re), lmks_shape)

def rotate(source, target, no_reflect=False):
    """Rotates the landmark set `source` so as to minimize its sum of squared interlandmark distances to `target`.

    Say X=`source` and Y=`target`. By default :func:`rotate` tries to find

    .. math:: \operatorname*{argmin}_{R \in O(k)} \| Y - XR \|^2

    That is, if `no_reflect` is `False`, :func:`rotate` might possibly reflect 
    X if it would achieve better alignment to Y. This behavior can be switched 
    off by setting `no_reflect` to `True`, in which case X will be aligned to Y 
    using a pure rotation :math:`R \in SO(k)`.

    Todo
    ----
    1. Handle when values are NaN.

    References
    ----------
    1. Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." no 3 (2017): 1-5. 
    `I found a pdf here <https://igl.ethz.ch/projects/ARAP/svd_rot.pdf>`_.

    Parameters
    ----------
    source : array-like
        A (p,k)-shaped landmark set corresponding to the source shape.

    target : array-like
        A (p,k)-shaped landmark set corresponding to the target shape.

    no_reflect : bool, optional
        Flag indicating whether the best alignment should exclude reflection 
        (default is False, which means reflection will be used if it achieves 
        better alignment).

    Returns
    -------
    result: dict
        aligned: numpy.ndarray
            A (p,k)-shaped landmark set consisting of the `source` landmarks 
            rotated to the `target`.
        
        R: numpy.ndarray
            A (k,k)-shaped array representing the right rotation matrix by 
            which `source` is rotated.
        
        D: numpy.ndarray
            A (k,)-shaped array representing the diagonal matrix of the SVD of np.dot(target.T, source).
    """
    result = {'aligned': None, 'R': None, 'D': None}
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
    result['aligned'] = np.matmul(source, R)
    result['R'] = R
    result['D'] = D
    return result

def opa(source, target, do_scaling=False, no_reflect=False):
    """Performs Ordinary Procrustes Alignment to transform the landmark set 
    `source` such that the squared Euclidean distance between `source` and 
    `target` is minimized.

    Say X=`source` and Y=`target` and `do_scaling` = `True`. 
    :func:`opa` tries to find

    .. math:: \operatorname*{argmin}_{\\beta > 0,\ R \in O(k),\ \gamma \in \mathbb{R}^k } D^2_{\mathtt{OPA}}(X, Y) = \| Y - \\beta X R - \mathbf{1_k} \gamma^T \|^2

    If `do_scaling` = `False`, :math:`\\beta = 1`. If `no_reflect` = `True`, 
    then just as in :func:`rotate`, :func:`opa` will force :math:`R \in SO(k)`.

    The Ordinary (Procrustes) Sum of Squares or OSS is defined as

    .. math:: OSS(X, Y) = \operatorname*{min}_{\\beta > 0,\ R \in O(k),\ \gamma \in \mathbb{R}^k } D^2_{\mathtt{OPA}}(X, Y)

    Note
    ----
    Generally for :func:`opa`, :math:`OSS(X1, X2) \\neq OSS(X2, X1)`.

    In contrast to :func:`opa`, :func:`gpa` is symmetric for the input matrices 
    in that :math:`G(X1, X2) = G(X2, X1)`.

    See Also
    --------
    rotate, gpa

    Parameters
    ----------
    source : array-like
        A (p,k)-shaped landmark set corresponding to the source shape.

    target : array-like
        A (p,k)-shaped landmark set corresponding to the target shape.

    do_scaling : bool, optional
        Flag indicating whether the best alignment should also find the optimal 
        :math:`\\beta` that minimizes :math:`D^2_{\mathtt{OPA}}`. The default 
        value of `do_scaling` is False, which means :math:`\\beta = 1`, or in 
        other words, `source` will not be scaled.

    no_reflect : bool, optional
        Flag indicating whether the best alignment should exclude reflection 
        (default is False, which means reflection will be used if it achieves 
        better alignment).

    Returns
    -------
    result: dict
        aligned: numpy.ndarray
            A (p,k)-shaped landmark set consisting of the `source` landmarks 
            aligned to the `target`.

        b: numpy.float64 or int
            A number representing the scaling factor :math:`\\beta` by which 
            `source` is scaled.
        
        R: numpy.ndarray
            A (k,k)-shaped array representing the right rotation matrix 
            :math:`R` by which `source` is rotated.
        
        c: numpy.ndarray
            A (k,)-shaped array representing the displacement :math:`\gamma` 
            between the centroids of `target` and the scaled+rotated `source`.
        
        oss: numpy.float64
            This number represents the Ordinary (Procrustes) Sum of Squares, 
            which is the minimum of :math:`D^2_{\mathtt{OPA}}`. Essentially, 
            the `oss` is the result of plugging in the optimal :math:`\\beta`, 
            :math:`R` and :math:`\gamma` into the :math:`D^2_{\mathtt{OPA}}` 
            objective.
        
        oss_stdized: numpy.float64
            This number is the Ordinary Sum of Squares `oss`, divided by the 
            squared norm of the centered target matrix. Loosely speaking it is 
            a kind of "normalization" or "relativization" of the disparity in 
            the `source` and `target` that is captured by the `oss`.

    Todo
    ----
    * Handle degenerate source, target landmarks.

    * Handle fewer landmarks in source.


    """
    result = { 'oss': None, 'oss_stdized': None, 'b': None, 'R': None, 'c': None, 'aligned': None }
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
        result['aligned'] = \
            remove_position(np.dot(Y0_norm*traceD*X0,result['R']), -muY)
    else:
        result['b'] = 1
        # The oss expression with a given b is 
        # ||Y||^2 + 2*b^2*||X||^2 - 2*b*||X||*||Y||*cos(rho(X,Y))
        # Again for standardized oss we divide by ||Y^2||.
        result['oss_stdized'] = 1 + (X0_ssq/Y0_ssq) - (2*(X0_norm/Y0_norm)*traceD)
        result['oss'] = Y0_ssq*result['oss_stdized']
        result['aligned'] = \
            remove_position(np.dot(X0_norm*X0,result['R']), -muY)
    # c is the gap between centroids of bXR and Y.
    result['c'] = muY - result['b']*np.dot(muX, result['R'])
    
    return result

def gpa(X, tol=1e-5,max_iters=10, do_project=False, do_scaling=False,
        no_reflect=False, unitize_mean=False):
    """Performs Generalized Procrustes Alignment to transform all the landmark 
    sets in `X` such that (a quantity proportional to) the sum of squared norms 
    of pairwise differences between all the landmark sets is minimized.

    Say :code:`len(X) = n`. :func:`gpa` tries to find

    .. math:: \operatorname*{argmin}_{\\beta_i > 0,\ R_i \in O(k),\ \gamma_i \in \mathbb{R}^k } g(X) = \\frac{1}{n} \sum_{i=1}^{n-1} { \sum_{j=i+1}^n {\| (\\beta_i X_i R_i + \mathbf{1_k} \gamma_i^T) - (\\beta_j X_j R_j + \mathbf{1_k} \gamma_j^T) \|^2}}

    The Generalized (Procrustes) Sum of Squares or G is defined as

    .. math:: G(X) = \operatorname*{inf}_{\\beta_i > 0,\ R_i \in O(k),\ \gamma_i \in \mathbb{R}^k } g(X)

    The GPA algorithm, per [drymar]_, tries to iteratively rotate and scale the 
    landmark sets in `X` until the sum of squared differences is below `tol`. 
    While the algorithm should converge quite fast, it can be forced to stop 
    the minimization loop after `max_iters` number of iterations.

    For an explanation of the other parameters, please see the Parameters 
    section.

    Note
    ----
    Re `do_project` and `do_scaling`: The projection used here is based on 
    [rohlf]_ and assumes that the aligned shapes are of unit centroid size, 
    which is not generally true when `do_scaling` is `True`. Consequently, if 
    both `do_project` and `do_scaling` are `True`, :func:`gpa` will issue a 
    warning, but proceed with the projection.

    Note
    ----
    Generally for :func:`opa`, :math:`OSS(X1, X2) \\neq OSS(X2, X1)`.

    In contrast to :func:`opa`, :func:`gpa` is symmetric for the input matrices 
    in that :math:`G(X1, X2) = G(X2, X1)`.

    See Also
    --------
    rotate, opa

    Parameters
    ----------
    X : array-like
        A (n,p,k)-shaped set of landmark sets that have to be aligned to each 
        other.

    tol : float, optional
        The sum of squared differences value that will be considered "low 
        enough" by the iterative rotation and scaling. The iterations will 
        continue until `tol` has been achieved or `max_iters` is reached, 
        whichever comes first.

    max_iters : int, optional
        The maximum number of iterations that the iterative rotation and 
        scaling is allowed to run for. The iterations will continue until `tol` 
        has been achieved or `max_iters` is reached, whichever comes first.

    do_scaling : bool, optional
        If `False`, :math:`\\beta_i = \\frac{1}{\| X'_i \|}`, where 
        :math:`X'_i` is the mean-centered :math:`X_i`. Else :math:`\\beta_i` is 
        calculated as per [tenb]_.

    do_project: bool, optional
        If `True`, the final aligned landmarks are orthogonally projected to 
        the tangent space at the mean of aligned landmark sets `mean`, 
        using equation 1 in [rohlf]_.

    no_reflect : bool, optional
        Flag indicating whether the best alignment should exclude reflection 
        (default is False, which means reflection will be used if it achieves 
        better alignment).

    unitize_mean: bool, optional
        Flag indicating whether the mean of aligned landmark sets `mean` 
        should be rescaled to have unit centroid size.

    Returns
    -------
    result: dict
        aligned: numpy.ndarray
            A (n,p,k)-shaped set of aligned landmark sets.
        
        mean: numpy.ndarray
            A (p,k)-shaped array representing the mean of the procrustes aligned landmark sets `aligned`.
        
        b: numpy.ndarray
            A (n,)-shaped array representing the scaling factor 
            :math:`\\beta_i` by which the centered :math:`X'_i` is scaled.
        
        ssq: numpy.float64
            This number represents the Generalized (Procrustes) Sum of Squares, 
            which is the infinimum of :math:`g`. Essentially, 
            the `ssq` is the result of plugging in the optimal :math:`\\beta_i`,
            :math:`R_i` and :math:`\gamma_i` into the :math:`g` objective.
        
    Warns
    -----
    UserWarning
        If both `do_project` and `do_scaling` are `True`

    References
    ----------
    .. [drymar] Dryden, I.L. and Mardia, K.V., 1998. Statistical shape analysis.
    .. [tenb] Ten Berge, J.M., 1977. Orthogonal Procrustes rotation for two or 
              more matrices. Psychometrika, 42(2), pp.267-276.
    .. [rohlf] Rohlf, F.J., 1999. Shape statistics: Procrustes superimpositions 
               and tangent spaces. Journal of Classification, 16(2), pp.197-223.

    Todo
    ----
    * Handle degenerate source, target landmarks.

    * Handle fewer landmarks in source.


    """
    res = {'aligned': None, 'mean': None, 'b': None, 'ssq': None}
    n_lmk_sets = lmk_util.num_lmk_sets(X)
    n_lmks = lmk_util.num_lmks(X)
    n_coords = lmk_util.num_coords(X)

    # 1. Remove position
    muX = get_position(X)
    X0 = remove_position(X, muX)
    
    # 2. Remove scale (if not do_scaling, we're just doing partial procrustes)
    X0_norm = get_scale(X0)
    X0 = remove_scale(X0, X0_norm)
    b = np.reciprocal(X0_norm)
    
    # 3. Rotate all lmk sets to the mean of all other lmk sets. Scale.
    aligned = X0
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
                all_but_i = aligned[all_i != i]
                mean_for_i = (1.0/(n_lmk_sets-1))*np.sum(all_but_i, axis=0)
                # Rotate all lmk sets to this mean
                aligned = rotate(aligned, mean_for_i, no_reflect)['aligned']
            ssq = get_ssqd(aligned)

        # 3.2. Scale
        if do_scaling:
            # We first get the biggest eigvec the nxn corr matrix.
            aligned_vecd = np.reshape(aligned, (n_lmk_sets, n_coords*n_lmks))
            X0_corrcoef = np.corrcoef(aligned_vecd)
            eig_vals, eig_vecs = np.linalg.eigh(X0_corrcoef)
            sort_perm = eig_vals.argsort()
            phi = eig_vecs[:, sort_perm][:, -1]
            if np.all(phi < 0):
                phi = np.abs(phi) # TODO: Is this okay to do?
            # The scale beta_i = sqrt(sum of sqd norms/ith sqd norm)*phi[i]
            aligned_norm = get_scale(aligned)
            aligned_ssq_norm = np.sqrt(np.sum(np.square(aligned)))
            frac = np.reciprocal(aligned_norm, dtype=np.float64)*aligned_ssq_norm
            beta = np.multiply(frac, phi)
            # Rescale aligned[i] by b_i
            aligned = np.multiply(aligned, np.reshape(beta, (n_lmk_sets,1,1)))
            # Update b
            b = np.multiply(b, beta)

        ssq = get_ssqd(aligned)
        curr_iter += 1
    
    print("ssq diff", ssq_old - ssq)

    # The mean is just the mean of the procrustes aligned lmk sets.
    mean = (1.0/n_lmk_sets)*np.sum(aligned, axis=0)
    if unitize_mean:
        mean = remove_scale(mean)

    if do_project:
        if do_scaling:
            w_msg = ("`do_project` assumes that the aligned lmk sets are scaled to have unit centroid size, which is not guaranteed if `do_scaling`. Proceeding with projection using the non-unit size lmk sets. See \'Rohlf, F. J. (1999). Shape statistics: Procrustes superimpositions and tangent spaces.\'")
            warnings.warn(w_msg)
        XC = mean.reshape((1, n_coords*n_lmks))
        X = aligned.reshape((n_lmk_sets, n_coords*n_lmks))
        # Get the projection matrix to project a shape onto X_c.
        XC_proj = (1.0/(XC @ XC.T)) * (XC.T @ XC)
        # Project all shapes onto the subspace orthogonal to X_c.
        X_tan = X @ (np.identity(n_coords*n_lmks) - XC_proj)
        # The above are like coordinates in the tangent space.
        # To get the "icons", we add back the mean.
        aligned = (X_tan + XC).reshape((n_lmk_sets, n_lmks, n_coords))
        # Recalculate the ssq
        ssq = get_ssqd(aligned)
        print("ssq diff", ssq_old - ssq)

    res['aligned'] = aligned
    res['mean'] = mean
    res['b'] = b
    res['ssq'] = ssq
    return res

def get_ssqd(X):
    """Alias for `lmk_util.ssqd(X)`.
    """
    return lmk_util.ssqd(X)

