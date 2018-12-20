import re
import pytest
import numpy as np
import morphops.lmk_util as lmk_util
from .helpers import make_haus, make_ngon

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