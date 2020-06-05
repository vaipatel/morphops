"""Provides IO functions to read from and write to files in common landmark data
file formats.
"""

import numpy as np
import morphops.lmk_util as lmk_util

class MopsFileReadError(Exception):
    pass

class MopsFileWriteError(Exception):
    pass

def read_dta(filename):
    """Reads \*.dta files, as written by the IDAV Landmark Editor.

    dta files typically have the following structure.

    1. Few comment lines. Comment lines start with a quotation mark (' or ").

    2. A header with structure "1 nL pk 1 9999 Dim=k". Here

       1. n is the number of specimens or number of landmark sets
       2. L in "nL" indicates that the file has specimen labels - assumed true
       3. p is the number of landmarks per landmark set
       4. k is the number of coordinates of each landmark (usually 2 or 3)

       The "1 9999" are ignored (but expected to exist) when reading. This is 
       because those two numbers are a misapplication of the NTS format, which 
       the DTA format is based on. Per the NTS format, the interpretation of 
       the "1 9999" is that the file has missing data indicated by 9999.
       DTA files always contain the "1 9999" numbers, regardless of whether the 
       file actually has missing data.

    3. n lines, each corresponding to the label of 1 specimen.
    4. n blocks of p lines. Each line contains k numbers. These correspond to 
       p k-D landmarks in each of the n specimens specified in the order of 
       appearance of their names in the preceding section.
    
    Todo
    ----
    Make implementation less non-pythonic if possible, while not allowing errors to go undetected.
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

def write_dta(filename, lmk_sets, names=[]):
    """Writes \*.dta files, as written by the IDAV Landmark Editor.

    See also
    --------
    read_dta: For an explanation of the \*.dta format.
    
    Todo
    ----
    Make implementation less non-pythonic if possible, without letting errors 
    slip through without raising.
    """
    n = lmk_util.num_lmk_sets(lmk_sets)
    p = lmk_util.num_lmks(lmk_sets)
    k = lmk_util.num_coords(lmk_sets)

    with open(filename, 'w+') as f:
        # Write some comments
        f.write("\'DTA file written by morphops\n")
        f.write("\n")
        # Write header
        header_els = [1, n, p*k, 1, 9999, 
                      ''.join(np.array(['Dim=',k]).astype(str))]
        header = ' '.join(np.array(header_els).astype(str))
        f.write(header + "\n")
        f.write("\n")
        # Write the names. Missing names are populated as 'InsertName{ID}', 
        # where ID goes from [len(names) + 1, n + 1).
        rem_names_ids = np.arange(len(names)+1,n+1).astype(str)
        rem_names = np.core.char.add('InsertName', rem_names_ids)
        names = np.append(names, rem_names)
        for name in names:
            f.write(name + "\n")
        
        # Write the coordinates
        for i in range(n):
            f.write("\n")
            for j in range(p):
                lmk = np.array(lmk_sets[i, j, 0:k])
                lmkstr = np.array2string(lmk, precision=20)
                lmkstr = lmkstr.replace('[','').replace(']','').strip()
                f.write(lmkstr + "\n")
        


            
            
