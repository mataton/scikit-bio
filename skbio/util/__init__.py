r"""Utilities for Developers (:mod:`skbio.util`)
============================================

.. currentmodule:: skbio.util

This package provides general exception/warning definitions used throughout
scikit-bio, as well as various utility functionality, including I/O and
unit-testing convenience functions.


Testing utilities
-----------------

Common functionality to support testing in skbio.

.. autosummary::
   :toctree: generated/

   get_data_path
   assert_ordination_results_equal
   assert_data_frame_almost_equal


Plotting utilities
------------------

.. autosummary::
   :toctree: generated/

   PlottableMixin


Miscellaneous utilities
-----------------------

Generally useful functionality that doesn't fit in more specific locations.

.. autosummary::
   :toctree: generated/

   get_rng
   cardinal_to_ordinal
   find_duplicates
   safe_md5
   classproperty


Developer warnings
------------------

.. autosummary::

   EfficiencyWarning


Details
-------

.. autoexception:: EfficiencyWarning


"""  # noqa: D412, D416, D205, D415

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._warning import EfficiencyWarning, SkbioWarning
from ._misc import cardinal_to_ordinal, find_duplicates, safe_md5, get_rng
from ._testing import (
    get_data_path,
    assert_ordination_results_equal,
    assert_data_frame_almost_equal,
    pytestrunner,
)
from ._decorator import classproperty
from ._plotting import PlottableMixin

__all__ = [
    "SkbioWarning",
    "EfficiencyWarning",
    "cardinal_to_ordinal",
    "find_duplicates",
    "safe_md5",
    "get_rng",
    "get_data_path",
    "assert_ordination_results_equal",
    "assert_data_frame_almost_equal",
    "classproperty",
    "pytestrunner",
    "PlottableMixin",
]
