.. image:: https://travis-ci.com/vaipatel/morphops.svg?branch=master
    :target: https://travis-ci.com/vaipatel/morphops

|

.. contents::
    :local:

|

Welcome to Morphops!
====================

Morphops implements common operations and algorithms for geometric
morphometrics, in python 3.

Some high-level operations in the current version are

* Centering, rescaling data: \
  :meth:`remove_position(lmk_sets) <morphops.procrustes.remove_position>`,
  :meth:`remove_scale(lmk_sets) <morphops.procrustes.remove_scale>`
* Rigid Rotation, Ordinary and Generalized Procrustes alignment: \
  :meth:`rotate(src_sets,tar_sets) <morphops.procrustes.rotate>`,
  :meth:`opa(src_set,tar_set) <morphops.procrustes.opa>`,
  :meth:`gpa(all_sets) <morphops.procrustes.gpa>`
* Thin-plate spline warping: \
  :meth:`tps_warp(X, Y, pts) <morphops.tps.tps_warp>`
* Reading from and writing to \*.dta files: \
  :meth:`read_dta(fn) <morphops.io.read_dta>`,
  :meth:`write_dta(fn,lmk_sets,names) <morphops.io.write_dta>`

Dependencies
------------

* numpy

Installation
------------

:code:`pip install morphops`

Usage Examples
--------------

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

   # Create a Thin-plate Spline warp from A to B and warp C.
   warped_C = mops.tps_warp(A, B, C)
   # warped_C contains the image of the pts in C under the TPS warp.
