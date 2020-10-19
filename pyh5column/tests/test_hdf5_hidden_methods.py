
from datetime import datetime
import numpy as np
import re
import os
import shutil
import tables as tb
import tempfile
import unittest
import uuid


from simpleh5 import H5ColStore


class TestCreateColumn(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _run_test(self, data, typecheck, size):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        with h.open() as h5:
            h._create_column(h5, '/data', data=data)

        with h.open() as h5:
            a = h5.get_node('/data')
            self.assertEqual(a.dtype, typecheck)
            self.assertEqual(len(a), size)

    def test_intcolumn_from_data(self):
        data = [1 for _ in range(1000)]
        self._run_test(data, 'int64', 1000)
        self._run_test(np.array(data), 'int64', 1000)

    def test_floatcolumn_from_data(self):
        data = [1.2 for _ in range(1000)]
        self._run_test(data, 'float64', 1000)
        self._run_test(np.array(data), 'float64', 1000)

    def test_strcolumn_from_data(self):
        data = ['a', 'bbb', 'cc']
        self._run_test(data, 'S3', 3)
        self._run_test(np.array(data), 'S3', 3)

    def test_column_matrix(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        data = np.random.rand(10, 3)
        with h.open(mode='a') as h5:
            h._create_column(h5, '/mytable/testcol', data=data)

        with h.open(mode='r') as h5:
            n = h5.get_node('/mytable/testcol')
            self.assertEqual(str(n.dtype), 'float64')
            self.assertTupleEqual(n.shape, (10, 3))


class TestConvertData(unittest.TestCase):

    def test_convert_str(self):

        h5 = H5ColStore('a.h5')
        d = h5._convert_data(['a', 'b', 'c'], 's2')
        self.assertIsInstance(d, np.ndarray)
        self.assertListEqual(d.tolist(), [b'a', b'b', b'c'])

    def test_convert_obj(self):
        h5 = H5ColStore('a.h5')
        d = h5._convert_data([['a'], ['b']], 'o10')
        self.assertIsInstance(d, np.ndarray)
        self.assertListEqual(d.tolist(), [b'\x91\xa1a1', b'\x91\xa1b1'])

    def test_convert_obj_compress(self):
        h5 = H5ColStore('a.h5')
        d = h5._convert_data([['a'], ['b']], 'c100')
        expect = [b'\x02\x01\x13\x08\x03\x00\x00\x00\x01\x00\x00\x00\x13\x00\x00\x00\x91\xa1a1',
                  b'\x02\x01\x13\x08\x03\x00\x00\x00\x01\x00\x00\x00\x13\x00\x00\x00\x91\xa1b1']
        self.assertIsInstance(d, np.ndarray)
        self.assertListEqual(d.tolist(), expect)

    def test_int(self):
        h5 = H5ColStore('a.h5')
        d = h5._convert_data([1, 2, 3], 'i')
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue(re.match(r'int', str(d.dtype)))
        self.assertListEqual(d.tolist(), [1, 2, 3])

    def test_float(self):
        h5 = H5ColStore('a.h5')
        d = h5._convert_data([1., 2., 3.], 'f')
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue(re.match(r'float', str(d.dtype)))
        self.assertListEqual(d.tolist(), [1., 2., 3.])
