

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
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION')) as fd:
    VERSION = fd.read().strip()

from simpleh5.utilities.serialize_utilities import msgpack_loads, msgpack_dumps


class TestMain(unittest.TestCase):

    def setUp(self):

        self.tmp_dir = tempfile.mkdtemp(dir='./')
        self.dint = [1 for _ in range(1000)]
        self.dfloat = [1.3 for _ in range(1000)]
        self.dstr = ['hello' for _ in range(1000)]
        self.dobj = [('yes', 1) for _ in range(1000)]
        self.dbytes = [b'yoloworld' for _ in range(1000)]
        self.dcomp = [('h' * 200) for _ in range(1000)]

        self.dt = {'t_int': 'i8', 't_float': 'n', 't_str': 'S100', 't_obj': 'o500', 't_bytes': 'o40', 't_comp': 'c100'}
        self.col_data = {'t_int': self.dint, 't_float': self.dfloat, 't_str': self.dstr, 't_obj': self.dobj,
                         't_bytes': self.dbytes, 't_comp': self.dcomp}

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_path(self):

        h5 = H5ColStore('abc.h5')
        self.assertEqual(h5._path('/hello', 'world'), '/hello/world')
        self.assertEqual(h5._path('hello', 'world'), '/hello/world')

        self.assertEqual(h5._path('/', 'world'), '/world')
        self.assertEqual(h5._path('/', '/world'), '/world')
        self.assertEqual(h5._path('', 'world'), '/world')

    def test_makedir_path(self):

        new_file = os.path.join(self.tmp_dir, 'level1', 'level2', 'level3', 'abc.h5')
        h5 = H5ColStore(new_file)
        h5.create_ctable('myoobj', col_dtypes={'col1': 'f'})
        self.assertTrue(os.path.exists(new_file))

    def test_matrix(self):

        new_file = os.path.join(self.tmp_dir, 'matrixtest.h5')
        h5 = H5ColStore(new_file)
        a = np.random.rand(2, 4)
        b = np.random.rand(3, 4)

        h5.create_ctable('/m1', col_dtypes={'col1': 'f'}, col_shapes={'col1': (0, 4)})

        h5.append_ctable('/m1', col_data={'col1': a}, col_dtypes={'col1': 'f'})
        h5.append_ctable('/m1', col_data={'col1': b}, col_dtypes={'col1': 'f'})
        x = h5.read_ctable('/m1')['col1']
        chk_mat = np.vstack((a, b))
        self.assertTrue(np.sum(np.sum(x - chk_mat)) == 0)

    def test_matrix_inds(self):

        new_file = os.path.join(self.tmp_dir, 'matrixtest.h5')
        h5 = H5ColStore(new_file)
        m = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        h5.append_ctable('/m1', col_data={'col1': m}, col_dtypes={'col1': 'f'})

        x = h5.read_ctable('/m1', inds=[1, 2])['col1']
        self.assertTrue(np.sum(np.sum(x - m[1:,:])) == 0)

    def test_matrix_initappend(self):

        new_file = os.path.join(self.tmp_dir, 'matrixtest.h5')
        h5 = H5ColStore(new_file)
        a = np.random.rand(2, 4)
        b = np.random.rand(3, 4)

        h5.append_ctable('/m1', col_data={'col1': a}, col_dtypes={'col1': 'f'})
        h5.append_ctable('/m1', col_data={'col1': b}, col_dtypes={'col1': 'f'})
        x = h5.read_ctable('/m1')['col1']
        chk_mat = np.vstack((a, b))
        self.assertTrue(np.sum(np.sum(x - chk_mat)) == 0)

    def test_msgpack_dump_load_list(self):

        d = ['abc', b'def', [1, 2, 3.4], {'hello': [1, 'world']}]

        comp = True
        r = msgpack_loads(msgpack_dumps(d, compress=comp), compress=comp, use_list=True)
        self.assertListEqual(d, r)

        comp = False
        r = msgpack_loads(msgpack_dumps(d, compress=comp), compress=comp, use_list=True)
        self.assertListEqual(d, r)

    def test_msgpack_dump_load_tuple(self):

        dresult = ('abc', b'def', (1, 2, 3.4), {'hello': (1, 'world')})
        d = ['abc', b'def', [1, 2, 3.4], {'hello': [1, 'world']}]

        comp = True
        r = msgpack_loads(msgpack_dumps(d, compress=comp), compress=comp, use_list=False)
        self.assertTupleEqual(dresult, r)

        comp = False
        r = msgpack_loads(msgpack_dumps(d, compress=comp), compress=comp, use_list=False)
        self.assertTupleEqual(dresult, r)

    def test_table_attrs_write(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.create_ctable('/table', col_dtypes=self.dt)
        attrs = h.table_info('/table')
        self.assertDictEqual(attrs['col_dtype'], self.dt)

        with h.open() as h5:
            h._write_attrs(h5, '/table', 'hello', 'world')
        attrs = h.table_info('/table')
        self.assertDictEqual(attrs['col_dtype'], self.dt)
        self.assertEqual(attrs['hello'], 'world')

        # overwrite
        with h.open() as h5:
            h._write_attrs(h5, '/table', 'hello', 'yolo')
        attrs = h.table_info('/table')
        self.assertSetEqual(set(attrs.keys()), {'col_dtype', 'hello', 'num_rows', '_version'})
        self.assertDictEqual(attrs['col_dtype'], self.dt)
        self.assertEqual(attrs['hello'], 'yolo')

        # get by name
        self.assertEqual(h.table_info('/table')['hello'], 'yolo')

    def test_table_all_vals(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.create_ctable('/table', col_dtypes=self.dt)

        # check read right after create
        b = h.read_ctable('/table')
        self.assertSetEqual(set(list(b.keys())), set(list(self.dt.keys())))

        h.append_ctable('/table', self.col_data)

        b = h.read_ctable('/table')
        for k, v in self.col_data.items():
            for i, x in enumerate(b[k]):
                self.assertEqual(x, v[i])

    def test_table_info(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        chk_dict = {'col_dtype': {'t_int': 'i', 't_float': 'f', 't_str': 's5',
                                  't_obj': 'o7', 't_bytes': 'o12', 't_comp': 's200'},
                    'col_flavor': {'t_int': 'python', 't_float': 'python', 't_str': 'python',
                                   't_obj': 'python', 't_bytes': 'python', 't_comp': 'python'},
                    'num_rows': 1000}
        info = h.table_info('/table')
        self.assertDictEqual(info['col_dtype'], chk_dict['col_dtype'])
        self.assertDictEqual(info['col_flavor'], chk_dict['col_flavor'])
        self.assertEqual(info['num_rows'], 1000)

    def test_list_return(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.create_ctable('/table', col_dtypes=self.dt)
        h.append_ctable('/table', self.col_data)

        simpah5_attrs = h.table_info('/table')
        self.assertSetEqual(set(simpah5_attrs['col_flavor'].values()), {'python'})

        b = h.read_ctable('/table')
        for k, v in b.items():
            self.assertIsInstance(v, list)

    def test_ndarray_return(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.create_ctable('/table', col_dtypes=self.dt)
        tmp_data = {}
        for k, v in self.col_data.items():
            tmp_data[k] = np.array(v)

        h.append_ctable('/table', tmp_data)

        b = h.read_ctable('/table')
        for k, v in b.items():
            # ignore packed columns
            if not re.match(r'[oc]', self.dt[k]):
                self.assertIsInstance(v, np.ndarray)
                
    def test_select_inds(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        tmp_data = {'c1': [], 'c2': []}
        for i in range(100):
            tmp_data['c1'].append(f'h{i}')
            tmp_data['c2'].append(i)
        
        h.append_ctable('/table', tmp_data)

        b = h.read_ctable('/table')
        self.assertDictEqual(tmp_data, b)

        b1 = h.read_ctable('/table', inds=list(range(10, 34)))
        self.assertListEqual(b1['c1'], tmp_data['c1'][10:34])
        self.assertListEqual(b1['c2'], tmp_data['c2'][10:34])
        
        # read outside range
        with self.assertRaises(IndexError) as e:
            b1 = h.read_ctable('/table', inds=list(range(90, 105)))

        # read out of order
        b1 = h.read_ctable('/table', inds=[10, 2, 78])
        self.assertDictEqual(b1, {'c2': [10, 2, 78], 'c1': ['h10', 'h2', 'h78']})

        # read limited columns
        b1 = h.read_ctable('/table', inds=[11, 3, 78], cols=['c2'])
        self.assertDictEqual(b1, {'c2': [11, 3, 78]})

    def test_table_nocomp_fail(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        dt = {'t_comp': 'o100'}
        h.create_ctable('/table', col_dtypes=dt)
        with self.assertRaises(Exception):
            h.append_ctable('/table', {'t_comp': self.dcomp}, resize=False)

    def test_table_safe_expand(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        dt = {'t_comp': 'o100'}
        h.create_ctable('/table', col_dtypes=dt)

        d1 = [('hello',), ('world', 0)]
        h.append_ctable('/table', {'t_comp': d1}, resize=True)
        # read data
        b = h.read_ctable('/table')
        self.assertListEqual(b['t_comp'], d1)
        # read info
        info = h.table_info('/table')
        self.assertEqual(info['col_dtype']['t_comp'], 'o100')

        # write data bigger than
        d2 = [('hello' * 50), '1']
        h.append_ctable('/table', {'t_comp': d2}, resize=True)
        b = h.read_ctable('/table')
        self.assertListEqual(b['t_comp'], d1 + d2)

        # ensure datatypes attribute was updated
        with h.open() as h5:
            node = h5.get_node('/table/t_comp')
            m = re.search(r'S(\d+)', str(node.dtype))
            new_len = int(m.group(1))
        info = h.table_info('/table')
        self.assertEqual(info['col_dtype']['t_comp'], f'o{new_len}')

    def test_append_without_create(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        info = h.table_info('/table')
        self.assertIn('col_dtype', info)
        self.assertIn('col_flavor', info)
        self.assertEqual(len(info['col_dtype']), 6)
        self.assertEqual(len(info['col_flavor']), 6)
        self.assertDictEqual(info['col_dtype'],
                             {'t_int': 'i', 't_float': 'f', 't_str': 's5',
                              't_obj': 'o7', 't_bytes': 'o12', 't_comp': 's200'})

        b = h.read_ctable('/table')
        for k, v in self.col_data.items():
            for i, x in enumerate(b[k]):
                self.assertEqual(x, v[i])

    def test_append_without_create_dtypes(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        info = h.table_info('/table')
        self.assertDictEqual(info['col_dtype'],
                             {'t_int': 'i', 't_float': 'f', 't_str': 's5',
                              't_obj': 'o7', 't_bytes': 'o12', 't_comp': 's200'})

        h.append_ctable('/table1', self.col_data, col_dtypes=self.dt)
        info = h.table_info('/table1')
        self.assertDictEqual(info['col_dtype'],
                             {'t_int': 'i8', 't_float': 'n', 't_str': 's100', 't_obj': 'o500', 't_bytes': 'o40',
                              't_comp': 'c100'})

        b = h.read_ctable('/table1')
        for k, v in self.col_data.items():
            for i, x in enumerate(b[k]):
                self.assertEqual(x, v[i])

    def test_addcol(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        with self.assertRaises(Exception):
            h.add_column('/table', 'new_col', [1 for _ in range(1000)])

        h.append_ctable('/table', {'init_col': [1 for _ in range(1000)]})

        with self.assertRaises(Exception):
            h.add_column('/table', 'new_col', [1 for _ in range(999)])

        info = h.table_info('/table')
        self.assertDictEqual(info, {'col_dtype': {'init_col': 'i'},
                                    'col_flavor': {'init_col': 'python'}, 'num_rows': 1000,
                                    '_version': VERSION})

        for col_name, col_data in self.col_data.items():
            h.add_column('/table', col_name, col_data)
        info = h.table_info('/table')

        chk = {'col_dtype': {'init_col': 'i', 't_int': 'i', 't_float': 'f', 't_str': 's5',
                             't_obj': 'o7', 't_bytes': 'o12', 't_comp': 's200'},
               'col_flavor': {'init_col': 'python', 't_int': 'python', 't_float': 'python', 't_str': 'python',
                              't_obj': 'python', 't_bytes': 'python', 't_comp': 'python'},
               'num_rows': 1000,
               '_version': VERSION}
        self.assertDictEqual(info, chk)

    def test_delcol(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        table_info = h.table_info('/table')
        self.assertDictEqual(table_info, {'col_dtype': {'col1': 'i', 'col2': 's1'},
                                          'col_flavor': {'col1': 'python', 'col2': 'python'},
                                          'num_rows': 3, '_version': VERSION})
        h.delete_column('/table', 'col1')
        table_info = h.table_info('/table')
        self.assertSequenceEqual(table_info, {'col_dtype': {'col2': 's1'},
                                              'col_flavor': {'col2': 'python'}, 'num_rows': 3, '_version': VERSION})

    def test_remove(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)
        _ = h.table_info('/table')
        h.delete_ctable('/table')
        info = h.table_info('/table')
        self.assertEqual(len(info), 0)

        # check for delete exceptions
        self.assertIsNone(h.delete_ctable('/table'))
        with self.assertRaises(tb.NoSuchNodeError):
            h.delete_ctable('/table', raise_exception=True)

    def test_col_same_length_exception(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        col_data = {
            'col1': list(range(10)),
            'col2': ['a', 'a', 'c', 'd', 'e', 'f', 'g', 'g', 'f']
        }
        with self.assertRaises(Exception):
            h.append_ctable('/table', col_data)

    def test_write_wrong_data(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        col_data = {
            'col1': [1, 2, 3],
            'col3': [[1, 2], ['3', '4'], [5, '6']],
            'col2': ['a', 'b', 'c'],
        }
        h.append_ctable('/table', col_data)
        info = h.table_info('/table')
        self.assertDictEqual(info['col_dtype'], {'col1': 'i', 'col2': 's1', 'col3': 'o6'})

        # setup incorrect data type on col2
        new_col_data = {
            'col1': [4.],
            # write incorrect data to col2
            'col2': [[5, ]],
            'col3': [[7, ]],
        }
        with self.assertRaises(Exception):
            h.append_ctable('/table', new_col_data)

        info = h.table_info('/table')
        data = h.read_ctable('/table')
        self.assertEqual(info['num_rows'], 3)
        self.assertListEqual(data['col1'], [1, 2, 3])
        self.assertListEqual(data['col2'], ['a', 'b', 'c'])
        self.assertListEqual(data['col3'], [(1, 2), ('3', '4'), (5, '6')])


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')
        self.col_data = {
            'col1': list(range(10)),
            'col2': ['a', 'a', 'c', 'd', 'e', 'f', 'g', 'g', 'f', 'c']
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _setup_get_data(self, match, in_kernel, cols=None):
        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        h.append_ctable('/table', self.col_data)
        return h.read_ctable('/table', query=match, cols=cols)

    def test_search_single(self):
        match = ['col1', '>', 2]
        b = self._setup_get_data(match, False)
        self.assertEqual(len(b['col1']), 7)
        self.assertEqual(len(b['col2']), 7)

    def test_search_single_ik(self):
        match = ['col1', '>', 2]
        b = self._setup_get_data(match, True)
        self.assertEqual(len(b['col1']), 7)
        self.assertEqual(len(b['col2']), 7)

    def test_search_or(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
        ]
        b = self._setup_get_data(match, False)
        self.assertEqual(len(b['col1']), 6)
        self.assertListEqual(b['col1'], [0, 1, 2, 3, 8, 9])
        self.assertEqual(len(b['col2']), 6)
        self.assertListEqual(b['col2'], ['a', 'a', 'c', 'd', 'f', 'c'])

    def test_search_or_ik(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
        ]
        b = self._setup_get_data(match, True)
        self.assertEqual(len(b['col1']), 6)
        self.assertListEqual(b['col1'], [0, 1, 2, 3, 8, 9])
        self.assertEqual(len(b['col2']), 6)
        self.assertListEqual(b['col2'], ['a', 'a', 'c', 'd', 'f', 'c'])

    def test_search_or_w_and(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
            ['col2', '==', 'a'],
            ['col1', '!=', 0],
        ]
        b = self._setup_get_data(match, False)
        self.assertEqual(len(b['col1']), 1)
        self.assertListEqual(b['col1'], [1])
        self.assertEqual(len(b['col2']), 1)
        self.assertListEqual(b['col2'], ['a'])

    def test_search_or_w_and_ik(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
            ['col2', '==', 'a'],
            ['col1', '!=', 0],
        ]
        b = self._setup_get_data(match, True)
        self.assertEqual(len(b['col1']), 1)
        self.assertListEqual(b['col1'], [1])
        self.assertEqual(len(b['col2']), 1)
        self.assertListEqual(b['col2'], ['a'])

    def test_search_or_w_and_restrict(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
            ['col2', '==', 'a'],
            ['col1', '!=', 0],
        ]
        b = self._setup_get_data(match, False, cols=['col2'])
        self.assertEqual(len(b['col2']), 1)
        self.assertListEqual(b['col2'], ['a'])
        self.assertSetEqual(set(b), {'col2'})

    def test_search_or_w_and_restrict_ik(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
            ['col2', '==', 'a'],
            ['col1', '!=', 0],
        ]
        b = self._setup_get_data(match, True, cols=['col2'])
        self.assertEqual(len(b['col2']), 1)
        self.assertListEqual(b['col2'], ['a'])
        self.assertSetEqual(set(b), {'col2'})

    def test_nomatch(self):
        match = [
            ['col1', '==', -9999],
        ]
        b = self._setup_get_data(match, True)
        self.assertListEqual(b['col1'], [])
        self.assertListEqual(b['col2'], [])


class TestDeleteRows(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')
        self.col_data = {
            'col1': list(range(10)),
            'col2': ['a', 'a', 'c', 'd', 'e', 'f', 'g', 'g', 'f', 'c']
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _check_match(self, match, in_kernel):
        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)
        h.delete_rows('/table', query=match)
        return h.read_ctable('/table')

    def test_match1(self):
        match = [
            [('col1', '>', 7), ('col1', '<=', 3)],
            ['col2', '==', 'a'],
        ]
        b = self._check_match(match, False)
        self.assertListEqual(b['col1'], [8, 9, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(b['col2'], ['f', 'c', 'c', 'd', 'e', 'f', 'g', 'g'])

        b = self._check_match(match, True)
        self.assertListEqual(b['col1'], [8, 9, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(b['col2'], ['f', 'c', 'c', 'd', 'e', 'f', 'g', 'g'])

    def test_delete_rows1(self):
        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        h.delete_rows('/table', rows=[2, 5, 6])
        b = h.read_ctable('/table')
        self.assertListEqual(b['col1'], [0, 1, 7, 3, 4, 8, 9])
        self.assertListEqual(b['col2'], ['a', 'a', 'g', 'd', 'e', 'f', 'c'])

        match = [
            ['col2', '==', 'a']
        ]
        h.delete_rows('/table', query=match)
        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': [8, 9, 7, 3, 4], 'col2': ['f', 'c', 'g', 'd', 'e']})

        match = [
            [('col2', '==', 'd'), ('col2', '==', 'g')]
        ]
        h.delete_rows('/table', query=match)
        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': [8, 9, 4], 'col2': ['f', 'c', 'e']})

    def test_delete_rows_no_match(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        match = ('col1', '==', 15)
        x = h.delete_rows('/table', query=match)
        self.assertIsNone(x)

    def test_delete_first_match(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', {'col1': ['abc', 'bde'], 'col2': [1, 2]})

        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': ['abc', 'bde'], 'col2': [1, 2]})

        match = [
            ['col1', '==', 'abc']
        ]
        h.delete_rows('/table', query=match)
        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': ['bde'], 'col2': [2]})

    def test_delete_single_item(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', {'col1': ['abc'], 'col2': [1]})

        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': ['abc'], 'col2': [1]})

        match = [
            ['col1', '==', 'abc']
        ]
        h.delete_rows('/table', query=match)
        b = h.read_ctable('/table')
        self.assertDictEqual(b, {'col1': [], 'col2': []})


class TestUpdateRows(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')
        self.col_data = {
            'col1': list(range(10)),
            'col2': ['a', 'a', 'c', 'd', 'e', 'f', 'g', 'g', 'f', 'c'],
            'col3': [('yolo', 'people') for _ in range(10)]
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_updates_single(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        match = [
            ['col2', '==', 'e'],
        ]
        new_data = {'col1': [20]}
        h.update_ctable('/table', match, new_data)
        b = h.read_ctable('/table')
        self.assertListEqual(b['col1'], [0, 1, 2, 3, 20, 5, 6, 7, 8, 9])

        h.update_ctable('/table', match, new_data)
        new_data = {'col1': [22]}
        h.update_ctable('/table', match, new_data)
        b = h.read_ctable('/table')
        self.assertListEqual(b['col1'], [0, 1, 2, 3, 22, 5, 6, 7, 8, 9])

        new_data = {'col3': [('sweet', 'now')]}
        check = [('yolo', 'people') for _ in range(10)]
        check[4] = ('sweet', 'now')
        h.update_ctable('/table', match, new_data)
        b = h.read_ctable('/table')
        self.assertListEqual(b['col3'], check)

    def test_updates_multi(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        match = [
            ['col2', '==', 'a'],
        ]
        new_data = {'col1': [20]}
        h.update_ctable('/table', match, new_data)
        b = h.read_ctable('/table')
        self.assertListEqual(b['col1'], [20, 20, 2, 3, 4, 5, 6, 7, 8, 9])

        new_data = {'col1': [21, 22], 'col2': ['w', 'z']}
        h.update_ctable('/table', match, new_data)
        b = h.read_ctable('/table')
        self.assertListEqual(b['col1'], [21, 22, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(b['col2'], ['w', 'z', 'c', 'd', 'e', 'f', 'g', 'g', 'f', 'c'])

    def test_updates_exception(self):

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.append_ctable('/table', self.col_data)

        match = [
            ['col2', '==', 'a'],
        ]
        new_data = {'col1': [21, 22, 23]}
        with self.assertRaises(Exception):
            h.update_ctable('/table', match, new_data)

        match = [
            [('col2', '==', 'a'), ('col2', '==', 'g')],
        ]
        new_data = {'col1': [21, 22]}
        with self.assertRaises(Exception):
            h.update_ctable('/table', match, new_data)


class TestDocStringExamples(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(dir='./')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_append(self):
        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))

        col_data = {
            'col1': [1, 2, 3],
            'col2': ['abc', 'def', 'geh'],
            'col3': [
                {'this is a dict': 123},
                {'a': 2, 'hello': 'world'},
                ['this is a list']
            ],
            'col4': np.array([1.2, 3.4, 5.6])
        }
        # define table
        col_dtype = {
            'col1': 'i',  # integers
            'col2': 's5',  # len 3 strings (after utf-8 conversion)
            'col3': 'o100',  # uncompressed objects with final byte len 100
            'col4': 'f'  # floats
        }
        h.create_ctable('mytable_1', col_dtypes=col_dtype)
        h.append_ctable('mytable_1', col_data)
        info = h.table_info('mytable_1')
        self.assertDictEqual(info,
                             {'col_dtype': {'col1': 'i', 'col2': 's5', 'col3': 'o100', 'col4': 'f'},
                              'col_flavor': {'col1': 'python', 'col2': 'python', 'col3': 'python', 'col4': 'numpy'},
                              'num_rows': 3, '_version': VERSION}
                             )

        h.append_ctable('/mytable_2', col_data)
        info = h.table_info('/mytable_2')
        self.assertDictEqual(info,
                             {'col_dtype': {'col1': 'i', 'col2': 's3', 'col3': 'o18', 'col4': 'f'},
                              'col_flavor': {'col1': 'python', 'col2': 'python', 'col3': 'python', 'col4': 'numpy'},
                              'num_rows': 3, '_version': VERSION}
                             )

    def test_flavor(self):
        # check ead of different flavors
        col_data = {
            'col1': [1, 2, 3],
            'col2': ['abc', 'def', 'geh'],
            'col3': [
                {'this is a dict': 123},
                {'a': 2, 'hello': 'world'},
                ['this is a list']
            ],
            'col4': np.array([1.2, 3.4, 5.6]),
            'col5': ['abc', 'def', 'geh'],
        }
        # define table
        col_dtype = {
            'col1': 'i',  # integers
            'col2': 's5',  # len 3 strings (after utf-8 conversion)
            'col3': 'o100',  # uncompressed objects with final byte len 100
            'col4': 'f',  # floats
            'col5': 'c100',  # uncompressed objects with final byte len 100
        }

        unique_filename = str(uuid.uuid4()) + '.h5'
        h = H5ColStore(os.path.join(self.tmp_dir, unique_filename))
        h.create_ctable('mytable_1', col_dtypes=col_dtype)
        h.append_ctable('mytable_1', col_data)

        data = h.read_ctable('mytable_1', flavor='python')
        for k, v in data.items():
            self.assertIsInstance(v, list)

        data = h.read_ctable('mytable_1', flavor='numpy')
        for k, v in data.items():
            if k == 'col3' or k == 'col5':
                self.assertIsInstance(v, list)
            else:
                self.assertIsInstance(v, np.ndarray)


if __name__ == "__main__":
    unittest.main()
