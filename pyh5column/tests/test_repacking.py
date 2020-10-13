"""
Added these tests after upgrading from pytables 3.4.2 -> 3.4.3
It seems that the file size on writes got significantly more efficient so that file size doesn't increase nearly as fast
"""

import numpy as np
import os
import shutil
import tempfile
import unittest
import uuid

from pyh5column import H5ColStore


class TestRepacking(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')

    def tearDown(self):
        pass
        shutil.rmtree(self.tmp_dir)

    def test_growth_size_limit(self):
        unique_filename = str(uuid.uuid4()) + '.h5'
        path_name = os.path.join(self.tmp_dir, unique_filename)
        h = H5ColStore(path_name)

        nloops = 200
        for x in range(nloops):
            data = {'t_comp': [1], 't_str': ['hello']}

            h.append_ctable('/table', col_data=data, resize=True)

            sz = os.stat(path_name).st_size

            # with pytables 4.3.2 the following grows to 28000 with nloops=200!!
            self.assertTrue(sz < 11000)

            # read data
            b = h.read_ctable('/table')
            self.assertListEqual(b['t_comp'], [1] * (x+1))
            self.assertListEqual(b['t_str'], ['hello'] * (x+1))

        h.repack()
        sz = os.stat(path_name).st_size
        self.assertTrue(sz < 11000)

    def test_growth_rand_repack(self):
        unique_filename = str(uuid.uuid4()) + '.h5'
        path_name = os.path.join(self.tmp_dir, unique_filename)
        h = H5ColStore(path_name)

        nloops = 200
        for x in range(nloops):
            news = ''.join([f'{np.random.randint(10)}'])
            data = {'t_comp': [np.random.rand()], 't_str': [news]}

            h.append_ctable('/table', col_data=data, resize=True)

            sz = os.stat(path_name).st_size

            # read data
            b = h.read_ctable('/table')
            self.assertEqual(len(b['t_comp']), x+1)
            self.assertEqual(len(b['t_str']), x+1)

        self.assertTrue(sz > 100000)
        h.repack()
        sz = os.stat(path_name).st_size
        self.assertTrue(sz < 15000)

    def test_repack_nonexist(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        path_name = os.path.join(self.tmp_dir, unique_filename)
        h = H5ColStore(path_name)
        with self.assertRaises(Exception):
            h.repack()


if __name__ == '__main__':
    unittest.main()
