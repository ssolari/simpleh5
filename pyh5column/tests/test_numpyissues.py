"""
Test issues related to numpy
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import unittest
import numpy as np
import time

import sys
PY_VER = sys.version[0]


class TestNumpyIssues(unittest.TestCase):

    def test_npstring_to_int(self):
        if PY_VER == '2':
            # in python2 the 'newint' dtype coming from builtins causes issues here
            # catch when this gets fixed
            with self.assertRaises(TypeError):
                s = int(np.string_('1'))
        else:
            self.assertEqual(1, int(np.string_('1')))

    def test_recarray_access(self):
        """
        Needed because numpy 1.10.0 and 1.10.1 are disturbingly slow due to recarray bug.
        So catch cases when something goes wrong
        :return:
        """
        ncols = 100
        names = ["c%d" % i for i in range(ncols)]
        formats = ["S5" for _ in range(ncols)]
        r = np.recarray((100, ncols), dtype=np.dtype({'names': names, 'formats': formats}))
        sttime = time.time()
        for j in np.random.randint(0, ncols, 1000):
            a = r["c%d" % j]
        # should run easily under 0.01 so 6x buffer
        self.assertLess(time.time()-sttime, 0.06)


if __name__ == "__main__":
    unittest.main()
