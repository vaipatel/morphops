.. image:: https://travis-ci.com/vaipatel/morphops.svg?branch=master
    :target: https://travis-ci.com/vaipatel/morphops

|

.. contents::
    :local:

|

Welcome to Morphops!
====================

Morphops implements common operations and algorithms for Geometric
Morphometrics, in Python 3.

Features
========

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
   res = mops.gpa([A, B, C])
   # res['aligned'] contains the aligned A, B, C. res['mean'] is their mean.

   # Create a Thin-plate Spline warp from A to B and warp C.
   warped_C = mops.tps_warp(A, B, C)
   # warped_C contains the image of the pts in C under the TPS warp.


What is Geometric Morphometrics?
================================

Geometric Morphometrics is a statistical toolkit for quantifying and studying
shapes of forms that are represented by homologous landmark sets.

"Shape" has a specific notion here. For a given landmark set, its shape refers
to the spatial information that survives after discarding its absolute
position, scale and rotation. So two landmark sets have the same shape if they
can be brought in perfect alignment by only changing their positions, scales
and rotations.

Common Operations and Algorithms in Studies
-------------------------------------------

Geometric Morphometrics is often used when pursuing statistical questions
involving the morphology of biological forms, like `do corvid species that 
frequently probe have longer bills and more to-the-side orbits than corvid species that frequently peck
<https://frontiersinzoology.biomedcentral.com/articles/10.1186/1742-9994-6-2>`_.
It helps inform the Data Collection, Preprocessing and Analysis
steps of such statistical studies with sound theoretical or practical justifications.

Data Collection
^^^^^^^^^^^^^^^

The most prevalent form of Data Collection involves picking homologous
landmarks on each form. For curving forms with few homologous points but
well-understood homologous regions, there is a notion of semilandmarks which
can "slide" to minimize equidistant sampling artifacts.

A common file format for saving landmarks for a set of specimens is the `*.dta`
format used by the IDAV Landmark Editor software.

Preprocessing
^^^^^^^^^^^^^

As discussed before, a central idea in Geometric Morphometrics is extracting
the "shapes" of the landmark sets. One way to achieve this is to use the
Generalized Procrustes Alignment algorithm or GPA. GPA aligns all the landmark
sets by modifying their locations, orientations and sizes so as to minimize
their collective interlandmark distances.

After this step, the aligned shapes all lie in a high-dimensional non-linear 
manifold. For example, if the orignal landmark sets were a set of triangles,
the aligned shapes lie on a sphere. Moreover, for naturally arising datasets,
the shapes likely lie very close to each other and are distributed around a
mean shape. This usually makes it permissible to project all the shapes into
the tangent space at the mean shape, and this way the final shape vectors lie
in a linear space.

Analysis
^^^^^^^^

With the shapes lying in a high-dimensional linear space after preprocessing,
they can now be submitted to various commonly used statistical procedures like
Principal Components Analysis and various kinds of regression for further
analysis.
