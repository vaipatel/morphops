import pytest
import numpy as np
import morphops.tps as tps
from .helpers import make_haus, make_ngon, get_2d_rot

pentagon = make_ngon(5)
pentagon_45 = make_ngon(5, np.pi/4.0)
fourgon = make_ngon(4)
fourgon_45 = make_ngon(4, np.pi/4.0)

K_matrix_2_data = [(np.zeros((5,2)), None, np.zeros((5,5)),
                  "X is (p,2) zeros. The result should be (p,p) zeros."),
                 (np.ones((5,2)), None, np.zeros((5,5)),
                 "X is (p,2) ones. The result should be (p,p) zeros."),
                 (fourgon, None, 
                 [[0, 2*np.log(2), 8*np.log(2), 2*np.log(2)],
                  [2*np.log(2), 0, 2*np.log(2), 8*np.log(2)],
                  [8*np.log(2), 2*np.log(2), 0, 2*np.log(2)],
                  [2*np.log(2), 8*np.log(2), 2*np.log(2), 0]],
                 "X is square of side sqrt(2). Result should be same as in 89 "
                 "paper, sec E.")]

K_matrix_3_data = [(np.column_stack((np.ones(4),fourgon)), None, 
                  [[0, np.sqrt(2), 2, np.sqrt(2)],
                   [np.sqrt(2), 0, np.sqrt(2), 2],
                   [2, np.sqrt(2), 0, np.sqrt(2)],
                   [np.sqrt(2), 2, np.sqrt(2), 0]],
                  "X is square of side sqrt(2). Result should be the distance "
                  "matrix.")]

tps_coefs_affine_data = [(fourgon, fourgon + 1, 
                np.zeros((4,2)), [[1,1],[1,0],[0,1]], "X is a square of side "
                "sqrt(2), Y = X + (1,1). Result should be that W is all "
                "zeros and A is a row of ones stacked on identity." 
                ),
                (fourgon, fourgon_45, 
                np.zeros((4,2)),np.row_stack(([0,0],get_2d_rot(-np.pi/4))),
                "X is a square of side sqrt(2), Y = XR. Result should be that "
                "W is all zeros and A is a row of zeros stacked on R.T." 
                )]

tps_warp_affine_data = [(fourgon, fourgon + 1, 
                fourgon, fourgon + 1, 
                "X is a square of side sqrt(2), Y = X + (1,1), pts = X. "
                "Result should Y."),
                (fourgon, fourgon_45, 
                fourgon, np.dot(fourgon, get_2d_rot(-np.pi/4)), 
                "X is a square of side sqrt(2), Y = XR, pts = square. Result "
                "should be pts*R.T." 
                )]

@pytest.mark.parametrize("X, Y, ans, scn", K_matrix_2_data)
def test_K_matrix_2(X, Y, ans, scn):
    print("K_matrix should evaluate the rbf at the distance between each "
          "lmk in X and Y when -", scn)
    assert np.allclose(tps.K_matrix(X,Y), ans)

@pytest.mark.parametrize("X, Y, ans, scn", K_matrix_3_data)
def test_K_matrix_3(X, Y, ans, scn):
    print("K_matrix should evaluate the rbf at the distance between each "
          "lmk in X and Y when -", scn)
    assert np.allclose(tps.K_matrix(X,Y), ans)

@pytest.mark.parametrize("X, Y, W_ans, A_ans, scn", tps_coefs_affine_data)
def test_tps_coefs(X, Y, W_ans, A_ans, scn):
    print("tps_coefs should evaluate spline weights W, A when -", scn)
    W, A = tps.tps_coefs(X, Y)
    assert np.allclose(W, W_ans)
    assert np.allclose(A, A_ans)

@pytest.mark.parametrize("X, Y, pts, warped_pts_ans, scn", tps_warp_affine_data)
def test_tps_warp(X, Y, pts, warped_pts_ans, scn):
    print("tps_warp should given warped pts when -", scn)
    assert np.allclose(tps.tps_warp(X,Y,pts), warped_pts_ans)
