import re
import pytest
import numpy as np
import morphops.procrustes as procrustes
from .helpers import make_haus, make_ngon, get_2d_refl

class TestProcrustesAlgos(object):

    # A weird house, plus its rotated, reflected, aligned versions.
    (haus,
    haus_c, haus0_b, haus0, two, haus0_scld, 
    haus0_Ro, haus0_Rf, haus0_rot, haus0_refl, 
    haus0_refl_al) = make_haus()

    rotate_data = [(haus0_rot, haus0, True, haus0, True,
                    "when X = YT, no_reflect, T rot, with 0 ssq"),
                   ([haus0_rot,haus0_rot], haus0,
                    True, [haus0, haus0], True,
                    "when 3-D X = YT, no_reflect, T rot, with 0 ssq"),
                   (haus0_refl, haus0, False, haus0, True,
                    "when X = YT, T refl, with 0 ssq"),
                   (haus0_refl, haus0, True, haus0_refl_al, False,
                    "when X = YT, no_reflect, T refl, with non-0 ssq")
                   ]
    
    opa_data = [(haus, haus0_rot, False, True,
                 haus0_rot, 1, haus0_Ro, -np.dot(haus_c, haus0_Ro), 0, 0,
                 "when X = YT + c, T rot, no_reflect, with 0 oss."),
                (haus, haus0_refl, False, False,
                 haus0_refl, 1, haus0_Rf, -np.dot(haus_c, haus0_Rf), 0, 0,
                 "when X = YT + c, T refl, with 0 oss."),
                (make_ngon(10)*3, make_ngon(10), False, False,
                 make_ngon(10)*3, 1, np.identity(2), 0, 10*np.square(3-1), 3+1,
                 "when X = bY, with 10*(b-1)^2 oss, 1 oss_stdized."),
                 (make_ngon(10)*3, make_ngon(10), True, False,
                 make_ngon(10), 1/3, np.identity(2), 0, 0, 0,
                 "when X = bY, do_scaling, with 0 oss.")]

    gpa_fail_data = [(np.random.randn(4,3), "when X is a 2d tensor"),
                     (np.random.randn(2,5,4,3), "when X is a 4d tensor")]

    # X, do_project, do_scaling, no_reflect, unitize_mean, mu, b
    gpa_data = [([haus0, haus0, haus0_scld], False, False, True, False,
                 haus0/haus0_b, [1.0/haus0_b, 1.0/haus0_b, 1.0/(two*haus0_b)],
                 "when X is scaled versions of the same haus0, no_reflect. "
                 "Result is: mu is haus0, b is 1/(scale of haus0)."),
                ([haus0, haus0, haus0_scld], False, True, True, True,
                 haus0/haus0_b, [1.0/haus0_b, 1.0/haus0_b, 1.0/(two*haus0_b)],
                 "when X is scaled versions of the same haus0, all opts true "
                 "except do_project. "
                 "Result is: mu is haus0, b is 1/(scale of haus0)."),
                ([haus0, haus0 @ get_2d_refl(0)],False, True, True, False,
                 (haus0 @ [[0,0],[0,1]])/haus0_b,
                 [1.0/haus0_b, 1.0/haus0_b],
                 "when X is haus0 and its y-refln, do_scaling, no_reflect. "
                 "Result is: mu is [(0,y_i)]*b, b is 1/(scale of haus0).")]

    @pytest.mark.parametrize("source, target, no_reflect, aligned, "
                             "should_match_target, scn",
                             rotate_data)
    def test_rotate(self, source, target, no_reflect, aligned, 
                    should_match_target, scn):
        print("rotate should solve argmin ||Y - XR||^2 -", scn)
        rot_res = procrustes.rotate(source, target, no_reflect)
        assert np.allclose(rot_res['aligned'], aligned)
        assert np.allclose(rot_res['aligned'], target) == should_match_target

    @pytest.mark.parametrize("source, target, do_scaling, no_reflect, aligned, "
                             "b, T, c, oss, oss_stdized, scn",
                             opa_data)
    def test_opa(self, source, target, do_scaling, no_reflect, aligned,
                 b, T, c, oss, oss_stdized, scn):
        print("opa should min ||Y - bXR + 1*c.T||^2 -", scn)
        opa_res = procrustes.opa(source, target, do_scaling, no_reflect)
        assert np.allclose(opa_res["aligned"], aligned)
        assert np.allclose(opa_res["b"], b)
        assert np.allclose(opa_res["R"], T)
        assert np.allclose(opa_res["c"], c)
        assert np.allclose(opa_res["oss"], oss)
        assert np.allclose(opa_res["oss_stdized"], oss_stdized)

    @pytest.mark.parametrize("X, scn", gpa_fail_data)
    def test_gpa_fail(self, X, scn):
        print("gpa should fail -", scn)
        matchstr = ("The input X must be a 3-D tensor of shape (n x p x k)")
        matchstr = re.escape(matchstr)
        with pytest.raises(ValueError,match=matchstr):
            procrustes.gpa(X)

    def test_gpa_warn(self):
        print("gpa should warn -", "when `do_scaling` and `do_project` "
              "are both `True`.")
        w_msg = ("`do_project` assumes that the aligned lmk sets are scaled to have unit centroid size, which is not guaranteed if `do_scaling`. Proceeding with projection using the non-unit size lmk sets. See \'Rohlf, F. J. (1999). Shape statistics: Procrustes superimpositions and tangent spaces.\'")
        matchstr = re.escape(w_msg)
        with pytest.warns(UserWarning,match=matchstr):
            X = [self.haus0, self.haus0_rot]
            procrustes.gpa(X,do_project=True, do_scaling=True)

    @pytest.mark.parametrize("X, do_project, do_scaling, no_reflect,"
                             "unitize_mean, mu, b, scn", gpa_data)
    def test_gpa(self, X, do_project, do_scaling, no_reflect, unitize_mean,
                 mu, b, scn):
        print("gpa should perform gpa -", scn)
        res = procrustes.gpa(X, do_project=do_project, do_scaling=do_scaling,
                             no_reflect=no_reflect, unitize_mean=unitize_mean)
        assert np.allclose(res['mean'], mu)
        assert np.allclose(res['b'], b)