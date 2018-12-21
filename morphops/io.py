import numpy as np

class MopsFileReadError(Exception):
    pass

def read_dta(filename):
    """Reads *.dta files in the written by IDAV Landmark Editor.

    R's Morpho pkg seems to have a much cooler implementation.
    """
    did_header = False
    names = []
    pts = []
    curr_line_i = -1
    with open(filename, 'r') as f:
        for line in f:
            curr_line_i += 1
            line = str.strip(line)
            # If line is empty or starts with quote, continue
            if len(line) == 0 or \
               line.startswith("\'") or line.startswith("\""):
                continue
            # If line indicates rectangular matrix and header not yet done,
            # we have a header. Eg- "1 2L 30 1 9999 Dim=3"
            if line.startswith("1") and not did_header:
                header_els = line.split()
                if len(header_els) is not 6:
                    raise MopsFileReadError("Error in line {}. A .dta file "
                    "header must have 6 parts.".format(curr_line_i))
                # Item 1 is the n_lmk_sets, followed by L or l
                n = int(header_els[1].replace('L','').replace('l',''))
                # Item 2 is the n_lmks*n_coords
                pk = int(header_els[2])
                # Item 5 contains the dimensions k.
                k = int((header_els[5]).lower().replace('dim=', ''))
                p = pk//k
                did_header = True
                continue
            
            # Read the lmk set names
            if len(names) < n:
                names.append(line)
                continue

            # Read in the coords
            coords = line.split()
            if len(coords) is not k:
                raise MopsFileReadError("Error in line {}. Could not parse the "
                "coordinates {}".format(curr_line_i, line))
            coords = np.array(coords).astype(np.float64)
            pts.append(coords)

    # Reshape into a n x p x k tensor
    lmk_sets = np.array(pts).reshape((n, p, k))
    return lmk_sets, names

            
            
