import re
import pytest
import numpy as np
import morphops.lmk_util as lmk_util
from .helpers import make_haus, make_ngon

num_lmk_sets_fail_data = [(make_ngon(4), "when X is a 2-D tensor"),
                         ([[make_ngon(4)]], "when X is a 4-D tensor")]

num_lmk_sets_data = [([[[]]], 1, "when X has a lmk set with no lmks, as 1."),
                     ([make_ngon(4), make_ngon(4)], 2, "when X has 2 lmk sets "
                     "as 2.")]

num_lmks_fail_data = [(1, "when X is a 0-D tensor"),
                      (np.arange(10), "when X is a 1-D tensor")]

num_lmks_data = [([[[]]], 1, "when X.shape is (1,1,0) as 1."),
                 (make_ngon(4), 4, "when X is a square as 4."),
                 ([make_ngon(4)], 4, "when X.shape is (1,4,2) as 4."),
                 ([[make_ngon(4)]], 4, "when X.shape is (1,1,4,2) as 4.")]

num_coords_fail_data = [(2, "when X is a 0-D tensor")]

num_coords_data = [([], 0, "when X.shape is (0,) as 0."),
                   (make_ngon(4), 2, "when X.shape is a planar square as 2."),
                   ([np.identity(3)], 3, "when X.shape is (1,3,3) as 3."),
                   (np.random.randn(1,1,6,4), 4, "when X.shape is (1,1,6,4) as "
                    "4.")]

ssqd_fail_data = [(make_ngon(4), "The input X must be a 3-D tensor "
                  "of shape (n x p x k)",
                  "when X is a 2D matrix"),
                  ([make_ngon(4)], "The input X must contain atleast "
                  "2 landmark sets.",
                  "when X is 3D with just one matrix")]

ssqd_data = [(np.zeros((3,4,2)), 0, "when X is all 0s, as 0."),
             ([np.identity(3), np.zeros((3,3))], 3.0/2,
              "when X is a 3x3 identity and zero, as 3."),
             ([make_ngon(4), make_ngon(4, np.pi/4)],
               4*(np.square(1 - np.sqrt(0.5)) + 0.5)/2,
               "when X is the unit square + it's pi/4 rotated form, as "
               "4*[(1-cos(pi/4))^2 + cos(pi/4)^2]")]

distance_matrix_data = [
        (np.zeros((4,2)), np.zeros((5,2)), np.zeros((4,5)),
        "when X is (p1,k) zeros and Y is (p2,k) zeros, as (pi,p2) zeros"),
        ([[0,0],[0,1]],[[1,1],[1,0]],[[np.sqrt(2),1],[1,np.sqrt(2)]],
        "when X and Y are oppo sides of a unit square, as 1s and sqrt(2)s")]

@pytest.mark.parametrize("X, scn", num_lmk_sets_fail_data)
def test_num_lmk_sets_fail(X, scn):
    print("num_lmk_sets should fail -", scn)
    err_msg = "The input X must be a 3-D tensor of shape (n x p x k)"
    with pytest.raises(ValueError,match=re.escape(err_msg)):
        lmk_util.num_lmk_sets(X)

@pytest.mark.parametrize("X, ans, scn", num_lmk_sets_data)
def test_num_lmk_sets(X, ans, scn):
    print("num_lmk_sets should give the number of lmk sets -", scn)
    assert lmk_util.num_lmk_sets(X) == ans

@pytest.mark.parametrize("X, scn", num_lmks_fail_data)
def test_num_lmks_fail(X, scn):
    print("num_lmks should fail -", scn)
    err_msg = "The input X must be a 2-D or 3-D tensor."
    with pytest.raises(ValueError,match=re.escape(err_msg)):
        lmk_util.num_lmks(X)

@pytest.mark.parametrize("X, ans, scn", num_lmks_data)
def test_num_lmks(X, ans, scn):
    print("num_lmks should give the number of lmks -", scn)
    assert lmk_util.num_lmks(X) == ans

@pytest.mark.parametrize("X, scn", num_coords_fail_data)
def test_num_coords_fail(X, scn):
    print("num_coords should fail -", scn)
    err_msg = "The input X must be a 1-D, 2-D or 3-D tensor."
    with pytest.raises(ValueError,match=re.escape(err_msg)):
        lmk_util.num_coords(X)

@pytest.mark.parametrize("X, ans, scn", num_coords_data)
def test_num_coords(X, ans, scn):
    print("num_coords should give the number of coords -", scn)
    assert lmk_util.num_coords(X) == ans

@pytest.mark.parametrize("X, err_msg, scn", ssqd_fail_data)
def test_ssqd_fail(X, err_msg, scn):
    print("ssqd should fail -", scn)
    with pytest.raises(ValueError,match=re.escape(err_msg)):
        lmk_util.ssqd(X)

@pytest.mark.parametrize("X, ans, scn", ssqd_data)
def test_ssqd(X, ans, scn):
    print("ssqd should give the sum of squared differences "
          "between all pairs of matrices -", scn)
    assert np.isclose(lmk_util.ssqd(X), ans)

@pytest.mark.parametrize("X, Y, ans, scn", distance_matrix_data)
def test_distance_matrix(X, Y, ans, scn):
    print("distance_matrix should give the distance between each lmk in X "
          "and Y when -", scn)
    assert np.allclose(lmk_util.distance_matrix(X,Y), ans)
    
