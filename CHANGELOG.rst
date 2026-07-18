=========
Changelog
=========

Morphops implements common operations and algorithms for Geometric
Morphometrics, in Python 3.

All notable changes to this project will be documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.1.0>`_.

Contributors to each release are listed in alphabetical order by first name. List
entries are sorted in descending chronological order.

Unreleased
==========

Changed
-------
- Dropped support for Python versions older than 3.12 and updated CI to test
  Python 3.12 through 3.14.

Fixed
-----
- Fixed compatibility with NumPy 2.x by replacing removed or deprecated NumPy
  APIs.
- Fixed invalid escape sequence warnings in math-heavy docstrings.

Added
-----
- all-contributors section to README. Added manually due to `lack of rst support in the cli, bot <https://github.com/all-contributors/all-contributors-cli/issues/300>`_, but see `PR 301 there <https://github.com/all-contributors/all-contributors-cli/pull/301>`_.

[0.1.13] - 2021-10-10
=====================

Added
-----
- This project now keeps a changelog.
