Welcome to Morphops!
====================

Morphops implements common operations and algorithms for Geometric
Morphometrics, in Python 3.

.. GitHub Actions
.. image:: https://github.com/vaipatel/morphops/actions/workflows/build.yml/badge.svg
    :target: https://github.com/vaipatel/morphops/actions/workflows/build.yml
    :alt: Build status

.. Read the Docs
.. image:: https://readthedocs.org/projects/morphops/badge/?version=latest
    :target: https://morphops.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. PyPI version
.. image:: https://img.shields.io/pypi/v/morphops
    :target: https://pypi.org/project/morphops
    :alt: PyPI version

Features
========

Some high-level operations in the current version are

* Centering, rescaling data:
* Rigid Rotation, Ordinary and Generalized Procrustes alignment:
* Thin-plate spline warping:
* Reading from and writing to \*.dta files:

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

Contributors ✨
===============

Thanks goes to these wonderful people (`emoji key <https://allcontributors.org/docs/en/emoji-key>`_):

.. raw:: html

    <!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
    <!-- prettier-ignore-start -->
    <!-- markdownlint-disable -->
    <table>
    <tr>
       <td align="center"><a href = "https://github.com/vaipatel"><img src="https://avatars.githubusercontent.com/u/6489594?v=4" width="100px;" alt=""/><br /><sub><b>Vaibhav Patel</b></sub></a></td>
       <td align="center"><a href="https://www.ntnu.edu/employees/hakon.w.anes"><img src="https://avatars.githubusercontent.com/u/12139781?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Håkon Wiik Ånes</b></sub></a><br /><a href="https://github.com/all-contributors/all-contributors/commits?author=hakonanes" title="Documentation">📖</a> <a href="#tool-hakonanes" title="Tools">🔧</a> <a href="#infra-hakonanes" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-hakonanes" title="Maintenance">🚧</a></td>
    </tr>
    </table>

    <!-- markdownlint-restore -->
    <!-- prettier-ignore-end -->

    <!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the `all-contributors <https://allcontributors.org>`_ specification.
Contributions of any kind are welcome!

This list is maintained manually until such time that the all-contributors bot supports rst. A possibly fix may be coming in `PR 301 there <https://github.com/all-contributors/all-contributors-cli/pull/301>`_.


|
| **(This file was autogenerated from README_for_docs.rst by running `make README_for_gh` in the docs directory)**
