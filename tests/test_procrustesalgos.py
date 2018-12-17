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

    @pytest.mark.parametrize("source, target, no_reflect, src_rot, "
                             "should_match_target, scn",
                             rotate_data)
    def test_rotate(self, source, target, no_reflect, src_rot, 
                    should_match_target, scn):
        print("rotate should solve argmin ||Y - XR||^2 -", scn)
        rot_res = procrustes.rotate(source, target, no_reflect)
        assert np.allclose(rot_res['src_rot'], src_rot)
        assert np.allclose(rot_res['src_rot'], target) == should_match_target

