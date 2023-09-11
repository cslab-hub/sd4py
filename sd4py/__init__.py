"""SD4Py

This module provides functions to discover subgroups on pandas DataFrames, and visualise them. It contains the following subpackages:

 * sd4py - Core functions to discover subgroups from a pandas DataFrame
 * sd4py_extra - Additional visualisation functions to display and further investigate results

 Note that, for convenience, `sd4py.sd4py' will also be imported as simple `sd4py', and `sd4py.sd4py_extra' will also be imported as `sd4py.extra'.
"""

# __all__ = ["sd4py", "sd4py_extra"]

from sd4py.sd4py import *
import sd4py.sd4py_extra
import sd4py.sd4py_extra as extra
