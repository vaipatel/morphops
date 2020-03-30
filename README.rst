.. image:: https://travis-ci.com/vaipatel/morphops.svg?branch=master
    :target: https://travis-ci.com/vaipatel/morphops

Welcome to Morphops!
====================

Morphops implements common operations and algorithms for 2d and 3d Geometric
Morphometrics, in Python 3.

What is Geometric Morphometrics?
======================

Geometric Morphometrics can be thought of as a quantitative study of shapes of
forms that are represented by homologous point sets or landmark sets. It
provides mathematical tools for spatially relativizing landmark sets - it helps
compare the spatial differences between equinumerous landmark sets that do arise
merely from differences in spatial sizes, locations and orientations of those
sets.

Features
========

Some high-level operations in the current version are

* Centering, rescaling data
* Rigid Rotation, Ordinary and Generalized Procrustes alignment
* Thin-plate spline warping
* Reading from and writing to \*.dta files

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

   # Perform Generalized Procrustes Alignment to align A, B, C.
   # :func:`gpa` is in the procrustes module.
   res = mops.gpa([A,B,C])
   # res['aligned'] contains the aligned A, B, C.
   # res['mean'] contains the mean of the aligned A, B, C.

   # Create a Thin-plate Spline warp from A to B and warp C.
   warped_C = mops.tps_warp(A, B, C)
   # warped_C contains the image of the pts in C under the TPS warp.
