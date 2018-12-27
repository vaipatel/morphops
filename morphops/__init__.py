"""The morphops module is the primary module of the Morphops library.

It contains implementations of common geometric morphometrics operations.
Some examples are - 

* IO operations to read/write landmark data

* Common preprocessing like procrustes alignment

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    lmk_util
    io
    procrustes
"""

# Not sure if prepending morphops to module is necessary, but makes it easier to import into jupyter for testing.
from .lmk_util import \
    transpose, num_coords, num_lmks, num_lmk_sets, ssqd
from .io import \
    MopsFileReadError, MopsFileWriteError, read_dta, write_dta
from .procrustes import \
    get_position, get_scale, remove_position, remove_scale, \
    rotate, opa, gpa
from ._version import __version__

VERSION = __version__
"""The version of this module."""