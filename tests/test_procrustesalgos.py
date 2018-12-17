import pytest
import numpy as np
import morphops.procrustes as procrustes
from .helpers import make_haus, make_ngon

class TestProcrustesAlgos(object):

    # A weird house, plus its rotated, reflected, aligned versions.
    (haus,
    haus_c, haus0, haus0_b, haus0_scld, 
    haus0_Ro, haus0_Rf, haus0_rot, haus0_refl, 
    haus0_refl_al) = make_haus()

    rotate_data = [(haus0_rot, haus0, True, haus0, True,
                    "when X = YT, no_reflect, T rot, with 0 ssq"),
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

    @pytest.mark.parametrize("source, target, no_reflect, src_rot, "
                             "should_match_target, scn",
                             rotate_data)
    def test_rotate(self, source, target, no_reflect, src_rot, 
                    should_match_target, scn):
        print("rotate should solve argmin ||Y - XR||^2 -", scn)
        rot_res = procrustes.rotate(source, target, no_reflect)
        assert np.allclose(rot_res['src_rot'], src_rot)
        assert np.allclose(rot_res['src_rot'], target) == should_match_target

    @pytest.mark.parametrize("source, target, do_scaling, no_reflect, src_rot, "
                             "b, T, c, oss, oss_stdized, scn",
                             opa_data)
    def test_opa(self, source, target, do_scaling, no_reflect, src_rot,
                 b, T, c, oss, oss_stdized, scn):
        print("opa should min ||Y - bXR + 1*c.T||^2 -", scn)
        opa_res = procrustes.opa(source, target, do_scaling, no_reflect)
        assert np.allclose(opa_res["src_rot"], src_rot)
        assert np.allclose(opa_res["b"], b)
        assert np.allclose(opa_res["R"], T)
        assert np.allclose(opa_res["c"], c)
        assert np.allclose(opa_res["oss"], oss)
        assert np.allclose(opa_res["oss_stdized"], oss_stdized)