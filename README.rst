Welcome to Morphops!
====================

Morphops implements common operations and algorithms for geometric
morphometrics, in python 3.

Dependencies
------------

* numpy

Installation
------------

:code:`pip install morphops`

Usage
-----

.. code-block:: python

   import morphops as mops
   # Create 3 landmark sets, each having 5 landmarks in 2 dimensions.
   A = [[0,0],[2,0],[2,2],[1,3],[0,2]]
   B = [[0.1,-0.1],[2,0],[2.3,1.8],[1,3],[0.4,2]]
   C = [[-0.1,-0.1],[2.1,0],[2,1.8],[0.9,3.1],[-0.4,2.1]]

   # Perform Generalized Procrustes alignment to align A, B, C.
   # :func:`gpa` is in the procrustes module.
   res = mops.gpa([A,B,C])
   
   # res['X0_ald'] contains the aligned A, B, C.
   # res['X0_mu'] contains the mean of the aligned A, B, C.
