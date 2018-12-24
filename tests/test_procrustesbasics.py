import pytest
import numpy as np
import morphops.procrustes as procrustes

class TestProcrustesBasics(object):

    M9_3x3 = np.arange(9).reshape((3,3))
    M9_3x3_c = np.array([3,4,5])
    M9_3x3_0 = np.array([[-3,-3,-3],[0,0,0],[3,3,3]])
    M12_2x3x2 = np.arange(12).reshape((2,3,2))
    M12_2x3x2_c = np.array([[2,3],[8,9]])
    M12_2x3x2_0 = np.tile([[-2,-2],[0,0],[2,2]], (2,1,1))
    M12_2x3x2_0n = np.tile([[-0.5,-0.5],[0,0],[0.5,0.5]], (2,1,1))

    get_position_fail_data = [(None, "None"),
                              (1.2, "0d tensor"),
                              ([], "Empty 1d tensor"),
                              ([1,2], "1d tensor"),
                              ([[[[]]]], "Empty 4d tensor"),
                              ([[[[1],[1]],[[1],[1]]]], "4d tensor")]

    get_position_data = [(M9_3x3, M9_3x3_c, "2d tensor"),
                         (M12_2x3x2, M12_2x3x2_c, "3d tensor")]

    rem_position_data = [(M9_3x3, None, M9_3x3_0, 
                         "2d tensor"),
                         (M12_2x3x2, M12_2x3x2_c, M12_2x3x2_0,
                         "3d tensor, explicit rem centroid"),
                         (M9_3x3 + [10,1,3], None, M9_3x3_0,
                         "2d tensor uniformly translated"),
                         (M9_3x3, 10, M9_3x3 - 10,
                         "2d tensor, rem 10")]

    get_scale_fail_data = [(None, "None"),
                           (1.2, "0d tensor"),
                           ([], "Empty 1d tensor"),
                           ([1,2], "1d tensor"),
                           ([[[[]]]], "Empty 4d tensor"),
                           ([[[[1],[1]],[[1],[1]]]], "4d tensor")]

    get_scale_data =  [(np.identity(3), np.sqrt(3), "2d identity"),
                       (M12_2x3x2_0, 4, "3d tensor")]

    rem_scale_data =  [(np.identity(3), None, np.identity(3)/np.sqrt(3),
                        "2d identity"),
                        (M12_2x3x2_0, [4, 4], M12_2x3x2_0n,
                        "3d tensor, explicit rem scale"),
                        (np.identity(3)*3, None, np.identity(3)/np.sqrt(3),
                        "2d identity uniformly scaled")]

    @pytest.mark.parametrize("lmks,scn", get_position_fail_data)
    def test_get_position_fail(self, lmks, scn):
        print("get_position should fail for -", scn)
        lmks_dim_size = len(np.shape(lmks))
        matchstr = ("Input lmks must have either 2 size dimensions for a "
                    "single specimen or 3 size dimensions for multiple "
                    "specimens. Instead got {}.").format(lmks_dim_size)
        with pytest.raises(ValueError,match=matchstr):
            centroid = procrustes.get_position(lmks)
            print(centroid)

    @pytest.mark.parametrize("lmks,centroid,scn", get_position_data)
    def test_get_position(self, lmks, centroid, scn):
        print("get_position should be the centroid/s for -", scn)
        assert(np.array_equal(procrustes.get_position(lmks), centroid))

    @pytest.mark.parametrize("lmks,position,lmks0,scn", rem_position_data)
    def test_remove_position(self, lmks, position, lmks0, scn):
        print("remove_position should mean center each set in -", scn)
        assert(np.array_equal(procrustes.remove_position(lmks,position), lmks0))

    def test_remove_and_get_position(self):
        print("get_position after remove_position should be the origin")
        lmks = np.random.randn(200,3)
        c = procrustes.get_position(procrustes.remove_position(lmks))
        assert(np.allclose(c, 0))
    
    @pytest.mark.parametrize("lmks,scn", get_scale_fail_data)
    def test_get_scale_fail(self,lmks,scn):
        print("get_scale should fail for -", scn)
        lmks_dim_size = len(np.shape(lmks))
        matchstr = ("Input lmks must have either 2 size dimensions for a "
                    "single specimen or 3 size dimensions for multiple "
                    "specimens. Instead got {}.").format(lmks_dim_size)
        with pytest.raises(ValueError,match=matchstr):
            centroid_size = procrustes.get_scale(lmks)
            print(centroid_size)

    @pytest.mark.parametrize("lmks,scale,scn", get_scale_data)
    def test_get_scale(self, lmks, scale, scn):
        print("get_scale should be the euclidean norm/s for -", scn)
        assert(np.allclose(procrustes.get_scale(lmks), scale))
    
    @pytest.mark.parametrize("lmks,scale,lmks_n,scn", rem_scale_data)
    def test_remove_scale(self, lmks, scale, lmks_n, scn):
        print("remove_scale should factor out scale in -", scn)
        assert(np.allclose(procrustes.remove_scale(lmks,scale), lmks_n))

    def test_remove_and_get_scale(self):
        print("get_scale after remove_scale should be 1")
        lmks = np.random.randn(200,3)
        s = procrustes.get_scale(procrustes.remove_scale(lmks))
        assert(np.allclose(s, 1))
