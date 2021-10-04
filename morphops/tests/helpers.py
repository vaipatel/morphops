import numpy as np

piOvr4 = np.pi/4

def get_2d_rot(angle=0):
    return np.array([[np.cos(angle),-np.sin(angle)],
                    [np.sin(angle),np.cos(angle)]])

def get_2d_refl(angle=0):
    return np.array([[-np.cos(angle),np.sin(angle)],
                    [np.sin(angle),np.cos(angle)]])

def make_ngon(nsides=6, angle_offset=0):
    ns = np.arange(0,2*np.pi, 2*np.pi/nsides) + angle_offset
    return np.column_stack((np.cos(ns),np.sin(ns)))

def make_haus():
    """Makes a weird house + its rotated, reflected, aligned versions.

    The roof of the house is tilted to the left, which creates asymmetry.
    """
    haus = np.row_stack((make_ngon(4, 3*piOvr4), 
                         make_ngon(3, -piOvr4/2) + [0, 1]))
    haus_c = np.array([0,3/7])
    haus0 = haus - haus_c
    haus0_b = np.sqrt(np.sum(np.square(haus0)))
    two = 2
    haus0_scld = two*haus0
    haus0_Ro = get_2d_rot(piOvr4)
    haus0_Rf = get_2d_refl(piOvr4)
    haus0_rot = np.dot(haus0, haus0_Ro)
    haus0_refl = np.dot(haus0, haus0_Rf)
    haus0_refl_al = np.dot(haus0, [[-1,0],[0,1]]) # haus0 refl across y axis
    return (haus,
            haus_c, haus0_b, haus0, two, haus0_scld, 
            haus0_Ro, haus0_Rf, haus0_rot, haus0_refl, 
            haus0_refl_al)