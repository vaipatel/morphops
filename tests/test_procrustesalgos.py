import pytest
import numpy as np
import morphops.procrustes as procrustes
from .helpers import make_haus

class TestProcrustesAlgos(object):

    # A weird house, plus its rotated, reflected, aligned versions.
    haus0, haus0_rot, haus0_refl, haus0_refl_al = make_haus()

    rotate_data = [(haus0_rot, haus0, False, haus0, True,
                    "rotated 2d matrices, perfect alignment"),
                   (haus0_refl, haus0, False, haus0, True,
                    "reflected 2d matrices, perfect alignment"),
                   (haus0_refl, haus0, True, haus0_refl_al, False,
                    "reflected 2d matrices, no_reflect, imperfect alignment")
                   ]

    @pytest.mark.parametrize("source, target, no_reflect, src_rot, "
                             "should_match_target, scn",
                             rotate_data)
    def test_rotate(self, source, target, no_reflect, src_rot, 
                    should_match_target, scn):
        print("rotate should rotate and/or reflect to least-squares align -",
              scn)
        rot_res = procrustes.rotate(source, target, no_reflect)
        assert np.allclose(rot_res['src_rot'], src_rot)
        assert np.allclose(rot_res['src_rot'], target) == should_match_target