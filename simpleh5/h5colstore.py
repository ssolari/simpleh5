"""
H5ColStore is the top-level object to use to write and read tabular column stores.   The H5ColStore object is passed a string
representing the full file path to the desired .h5(hdf5) file.   Instantiating the object is harmless, all future operations
will operate on this .h5 file.

Example::

    my_column_store = H5ColStore('all_tables.h5')

The goal of H5ColStore is to create a simplified experience with high performance for data scientists to focus on
analytics at the trade off of 'optimal' performance.   Time and effort are a trade off and H5ColStore tries to enable
a focus on using data in an efficient manner rather than providing hooks into all parameterization of how a file can
be stored and what compressors are used.
"""

from datetime import datetime
import logging
import numpy as np
import os
import re
import tables as tb
import traceback
from typing import Optional, Union
import uuid

from simpleh5.utilities.search_utilities import _filter_inds
from simpleh5.utilities.serialize_utilities import msgpack_dumps, msgpack_loads

ATTR_COLDTYPE = 'col_dtype'
ATTR_COLFLAV = 'col_flavor'
ATTR_NROWS = 'num_rows'
ATTR_CTABLE = '_ctable_attrs'
with open(os.path.join(os.path.dirname(__file__), '..', 'VERSION')) as fd:
    VERSION = fd.read().strip()


class H5ColStore(object):

    def __init__(self, h5file: str):

        if not re.search('\.h5$', h5file):
            raise Exception(f'h5file should have a .h5 extension')

        # path to the .h5 file that will be operated on
        self._h5file = h5file

        # default all operations to be blosc filter with highest comp level
        self._filters = tb.Filters(complevel=9, complib='blosc', fletcher32=False)

    def __str__(self):

        with self.open(mode='r') as h5:
            return h5.__str__()

    def open(self, mode: str='a') -> tb.File:
        """
        Open the file and return file handle to use with methods requiring open file handle.

        Should be used as::

            colstore = SimpaHdf5(filename)
            with colstore.open(mode='a') as h5:
                colstore.fh_somemethod(h5, ...)

        :param mode:  open mode in ['r', 'r+', 'a', 'w'] (default='a')

        :return:
        """

        if mode == 'a' or mode == 'w':
            try:
                fh = tb.open_file(self._h5file, mode=mode, filters=self._filters)
            except IOError:
                os.makedirs(os.path.dirname(self._h5file), exist_ok=True)
                fh = tb.open_file(self._h5file, mode=mode, filters=self._filters)
        else:
            fh = tb.open_file(self._h5file, mode=mode, filters=self._filters)

        return fh

    def create_ctable(self, table_name: str, col_dtypes: dict, col_shapes: Optional[dict]=None,
                      expectedrows: int=10000) -> None:
        """
        Create a new coltable under table path.  Note that all data is compressed therefore unknown length strings can
        be hedged by using a large number and allowing compression to save space.   Even with S1000 a string 'a' will
        take up a much smaller amount of space due to compression.  Same with compressed objects which are actually
        compressed twice to maximize ability to not overflow specified length in unknown cases.

        :param table_name: name of columnar table
        :param col_dtypes: dictionary of (column-name -> dtype) arguments.  Dtype(str) should be one of:

            * 'i': 64-bit integer
            * 'f': 64-bit float
            * 'n': 64-bit float
            * 'sx': len x string (unicode will be converted to utf-8 so length applies to final utf-8 strings)
            * 'ox': len x object (objects are serialized via msgpack so len refers to final serialized bytes len +1)
            * 'cx': len x compressed object (msgpack obj is compressed so len refers to final compressed len)
                Note small objects when compressed can increase in size.
        :param col_shapes: (optional) dictionary of shapes for each column.  Should have the form (0,) or (0, x).
            Mostly used when a 2-d array will be a column.  Not needed if a column is 1 dimension i.e. (0,).  Note the
            first entry is 0 if creating an empty table and appending data.
        :param expectedrows: (optional, default=10000) determines chunk size, if number of rows will be very large,
            then good to specify to optimize chunking.

        :return:

        Example table creation from definition::

            h = SimpaHdf5('myfile.h5')

            # define table
            col_dtype = {
                'col1': 'i'  # integers
                'col2': 's3'  # len 3 strings (after utf-8 conversion)
                'col3': 'o100'  # uncompressed serialized objects with final byte len 100
                'col4': 'f'  # floats
                'col5': 'c200'  # compressed serialized objects with final byte len 200
            }
            h.create_table('table1', col_dtype)


        """
        if col_shapes is None:
            col_shapes = {}

        with self.open(mode='a') as h5:
            self._create_ctable(h5, table_name, col_dtypes, col_shapes, expectedrows=expectedrows)

    def _create_ctable(self, h5, table_path, col_dtype, col_shapes, expectedrows: int=10000):

        # convert all dtypes to lowercase
        for col in col_dtype:
            col_dtype[col] = col_dtype[col].lower()

        for col_name, dtype in col_dtype.items():
            self._create_column_from_dtype(h5, table_path, col_name, dtype, col_shapes.get(col_name, None),
                                           expectedrows=expectedrows)

        self._write_attrs(h5, table_path, ATTR_COLDTYPE, col_dtype)

    def delete_ctable(self, table_name: str, raise_exception=False) -> None:
        """
        Delete the table if it exists.  Note that this will not reduce the size of the file.  A repack is needed
        to reduce file size after data deletion.

        :param table_name: path to table
        :param raise_exception: raise an exception if the table doesn't exist. If False and the table does not exist
            the method will simply return.
        :return:
        """
        nodepath = self._norm_path(table_name)
        with self.open(mode='a') as h5:
            if raise_exception:
                h5.remove_node(nodepath, recursive=True)
            else:
                try:
                    h5.remove_node(nodepath, recursive=True)
                except:
                    pass

    def add_column(self, table_path: str, col_name: str, col_data: (list, tuple, np.ndarray)=None,
                   col_dtype: str=None, shape=(0,)):
        """
        Add a column of data to an existing table.  The length of the data must be the same length as the existing
        table.  The addition of a column is a cheap operation since data is stored by column.

        :param table_path: internal path to table
        :param col_name: name of new column
        :param col_data: data for new column
        :param col_dtype: optional data type
        :param shape: shape of column if different than (0,)
        :return:
        """

        if col_data is None:
            raise NotImplementedError(f'Default back filling is coming soon...')

        with self.open(mode='a') as h5:

            table_info = self._table_info(h5, table_path)
            if len(table_info) == 0:
                raise Exception(f'Addcol needs an existing table at {table_path} in {self._h5file}')

            if table_info[ATTR_NROWS] != len(col_data):
                raise Exception(f'New column data different length ({len(col_data)}) '
                                f'than table ({table_info[ATTR_NROWS]}) in {self._h5file}')

            if col_dtype:
                self._create_column_from_dtype(h5, table_path, col_name, col_dtype, shape)
                self._add_column(h5, table_path, col_name, col_data, col_dtype, True)
            else:

                col_dtype = self._create_column_from_data(h5, table_path, col_name, col_data)

            # set the datatype
            coldt = table_info[ATTR_COLDTYPE]
            coldt[col_name] = col_dtype
            self._write_attrs(h5, table_path, ATTR_COLDTYPE, coldt)

            # set the flavor
            colflavor = table_info[ATTR_COLFLAV]
            if isinstance(col_data, np.ndarray):
                colflavor[col_name] = 'numpy'
            else:
                colflavor[col_name] = 'python'
            self._write_attrs(h5, table_path, ATTR_COLFLAV, colflavor)

    def _add_column(self, h5: tb.File, table_name: str, column_name: str, data: list, dtype: str, resize: bool):

        data = self._convert_data(data, dtype)
        colpath = self._path(table_name, column_name)

        if re.match(r'[osc](\d+)', dtype):
            colnode = self._safe_col_str_change(h5, colpath, dtype, data, resize)
        else:
            colnode = self._get_node(h5, colpath)
            if colnode is None:
                raise Exception(f"Table column doesn't exist: {colpath} in {self._h5file}")

        colnode.append(data)

    def delete_column(self, table_name: str, col_name: str, raise_exception=False) -> None:
        """
        Delete a single column from a table.

        :param table_name: table path
        :param col_name: name of column
        :param raise_exception: ignore exceptions when trying to delete colom (like for a non-existant column)
        :return:
        """

        colpath = self._path(table_name, col_name)
        with self.open(mode='a') as h5:

            simpah5_attrs = self._read_attrs(h5, table_name)
            if not simpah5_attrs and raise_exception:
                raise Exception(f"Table doesn't exist: {table_name} in {self._h5file}")
            elif col_name not in simpah5_attrs[ATTR_COLDTYPE] and raise_exception:
                raise Exception(f"Column doesn't exist: {table_name}/{col_name} in {self._h5file}")

            if raise_exception:
                h5.remove_node(colpath)
                del simpah5_attrs[ATTR_COLDTYPE][col_name]
                del simpah5_attrs[ATTR_COLFLAV][col_name]
            else:
                try:
                    h5.remove_node(colpath)
                    del simpah5_attrs[ATTR_COLDTYPE][col_name]
                    del simpah5_attrs[ATTR_COLFLAV][col_name]
                except:
                    pass
            if simpah5_attrs:
                self._write_attrs(h5, table_name, ATTR_COLDTYPE, simpah5_attrs[ATTR_COLDTYPE])
                self._write_attrs(h5, table_name, ATTR_COLFLAV, simpah5_attrs[ATTR_COLFLAV])

    def append_ctable(self, table_name: str, col_data: dict, col_dtypes: Optional[dict]=None,
                      resize: bool=True) -> None:
        """
        Append data to all columns in table.  Data for columns must all be the same length and all columns must
        be specified.

        See also :meth:`.create_ctable` for table data types and creating tables.

        If the table does not exist it will be created with the datatypes defined by the data.
        Consider creating the table first with predefined data types, or specifying the col_dtypes attribute, which
        will set the datatypes for the table on the first write only.

        On the first append of any data the flavor (python or numpy) of each data column is
        inspected (from the first element) and the flavor correspondingly stored with the table.

        On subsequent reads of the data the stored flavor (list or ndarray) of data will be returned.
        If the data is passed as a numpy ndarray then the data will be returned as a ndarray if passed as a list then
        data will be returned as a python list.

        'o' or 'c' datatypes always have the flavor of a python list.

        :param table_name: name of table to write
        :param col_data: dictionary of {'column_name': [data1, data2, ...], ...}
        :param col_dtypes: dictionary of {'column_name': dtype1, ...}.  col_dtypes is only
            used if table does not exist on first append and will make call to :meth:`.create_ctable`.
        :param resize: Applies to string columns [s, o, c]. (default=True).
            If True, prevents possibility of overflowing string or object out of defined column size.
            In the case that a string object is too large, the entire column is
            rewritten to the size of the largest string in the newly appended data.  Note, this can be expensive!
            The entire column is read into memory before rewriting.
            If False an exception will be raised if any data is longer than the existing size.

        :return: None

        Example data::

            h = SimpaHdf5('myfile.h5')

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

        Create then append data::

            # define table
            col_dtype = {
                'col1': 'i',  # integers
                'col2': 's5',  # len 3 strings (after utf-8 conversion)
                'col3': 'o100',  # uncompressed objects with final byte len 100
                'col4': 'f'  # floats
            }

            h.create_ctable('mytable_1', col_dtype)
            h.append_ctable('mytable_1', col_data)

        mytable_1 will have the following dtype::

            info = h.table_info('mytable_2')
            print(info)
            # {'col_dtype': {'col1': 'i', 'col2': 's5', 'col3': 'o100', 'col4': 'f'},
            #  'col_flavor': {'col1': 'python', 'col2': 'python', 'col3': 'python', 'col4': 'numpy'},
            #  'num_rows': 3}

        OR create from data alone::

            h.append_ctable('mytable_2', col_data)

        mytable_2 will have the following dtype::

            info = h.table_info('mytable_2')
            print(info)
            # {'col_dtype': {'col1': 'i', 'col2': 's3', 'col3': 'o18', 'col4': 'f'},
            #  'col_flavor': {'col1': 'python', 'col2': 'python', 'col3': 'python', 'col4': 'numpy'},
            #  'num_rows': 3}

        """
        if col_dtypes is None:
            col_dtypes = {}

        with self.open(mode='a') as h5:
            self._append_ctable(h5, table_name, col_data, resize=resize, col_dtypes=col_dtypes)

    def update_ctable(self, table_name: str, query: Union[list, tuple], col_data: dict,
                      resize: bool=True) -> None:
        """
        Perform an update of data in an existing table.

        :param table_name: table to update
        :param query: (required) match criteria to define rows to update
        :param col_data: (required) data to replace in each column.
            Only columns and data in col_data will be updated.
            col_data should either contain lists of length 1 (all found rows will be updated with this information),
            or the exact same length as the number of indices found. If data is presented with len > 1 and a mismatch
            occurs with the length of the indices that are returned from the match, an Exception will be raised.
        :param resize: True (default) prevents possibility of overflowing object or string columns.
            In the case that an object is too large the entire column is
            rewritten to the size of the largest string in the new append.  Note the entire column
            is read into memory before rewriting.  If False an exception will be raised, but other columns may have
            been written.
        :return:
        """

        if not query:
            raise Exception(f"A query is required for update_ctable in {self._h5file}")

        # error checking
        assert len(col_data) > 0
        change_len = None
        for col in col_data:
            if change_len is None:
                change_len = len(col_data[col])
            elif change_len != len(col_data[col]):
                raise Exception(f"Column data specified is not the same length col {col} = {len(col_data[col])} "
                                f"in {self._h5file}")

        with self.open(mode='a') as h5:

            simpah5_attrs = self._read_attrs(h5, table_name)
            col_dtype = simpah5_attrs[ATTR_COLDTYPE]

            # return a dictionary of node name to node pointers
            node_data = h5.get_node(self._norm_path(table_name))._v_children
            inds = _filter_inds(node_data, query)

            inds = np.nonzero(inds)[0]
            # check that inds match column change length
            if change_len != 1 and change_len != len(inds):
                raise Exception(f"Update column length different than rows length found in match in {self._h5file}")

            for col, data in col_data.items():

                if col not in col_dtype:
                    logging.warning(f"Column {col} specified for updating, "
                                    f"but not found in table {table_name}...ignoring")

                data = self._convert_data(col_data[col], col_dtype[col])
                colpath = self._path(table_name, col)

                if re.match(r'[osc](\d+)', col_dtype[col]):
                    colnode = self._safe_col_str_change(h5, colpath, col_dtype[col], data, resize)
                else:
                    colnode = self._get_node(h5, colpath)
                    if colnode is None:
                        raise Exception(f"Table column doesn't exist: {colpath} in {self._h5file}")

                colnode[inds] = data

    def read_ctable(self, table_name: str, cols: Optional[list]=None, query: Union[list, tuple]=(),
                    inds: Union[list, tuple]=(), flavor: str='') -> dict:
        """
        Read data from the table and return a dictionary of columns.  Data is returned in the flavor it was stored in.

        Search capability exists through query parameter.  Both old style and new style work (see below).

        Example read::

            h = SimpaHdf5('myfile.h5')
            data = h.read_ctable('table1', cols=['col2', 'col4'])

            # return data will look like
            data = {
                'col2': [c2_val1, c2_val2, ...],
                'col4': [c4_val1, c4_val2, ...]
            }

        query can filter datavalues across all columns even if they are not returned. A query has the format::

            # (col1 > 1.2) & (col1 < 2.3) & (col2 == 'abc')
            query = [
                ('col1', '>', 1.2),
                ('col1', '<', 2.3),
                ('col2', '==', 'abc')
            ]

            # col1 == 1.2
            query = ('col1', '==', 1.2)
            # NOTE: in the case of a single condition as above the additional surrounding list is not needed.

            # (col1 > 1.2) & (col1 < 2.3) & ((col2 == 'abc') | (col2 == 'def') | (col3 == True)) & (col2 != 'ggg')
            query = [
                ('col1', '>', 1.2),
                ('col1', '<', 2.3),
                (('col2', '==', 'abc'), ('col2', '==', 'def'), ('col3', '==', True)),
                ('col2', '!=', 'ggg')
            ]

        Where each column condition in query list are logical & ('and') together.  Sub-conditions within a single
        query column name are logical | ('or') together.

        :param table_name: path to table
        :param cols: return only this list of columns
        :param query: see description above
        :param inds: a list of indices to pull, will override any query
        :param flavor: ['python', 'numpy'], if set will force columns to be returned as either lists(python) or
            numpy arrays (numpy).   Columns specified as objects 'o' or compressed objects 'c' ignore flavor.
            Returning numpy arrays can be slightly faster.
        :return: return a dictionary of column name to list/array values
        """

        return_data = None
        col_dtype = None

        with self.open(mode='r') as h5:

            simpah5_attrs = self._read_attrs(h5, table_name)
            if len(simpah5_attrs) == 0:
                raise Exception(f"{table_name} doesn't exist in {self._h5file}")
            col_dtype = simpah5_attrs[ATTR_COLDTYPE]

            # determine columns to return
            return_cols = set(col_dtype.keys())
            if cols:
                return_cols = return_cols.intersection(set(cols))

            if query and not inds:
                allcols = h5.get_node(self._norm_path(table_name))._v_children
                inds = _filter_inds(allcols, query)

            # get data off disk
            return_data = {}
            for col in return_cols:

                node = self._get_col(h5, self._path(table_name, col))
                if len(inds) == 0:
                    # read in all node data
                    return_data[col] = node.read()
                else:
                    if len(node.shape) == 2:
                        return_data[col] = node[np.array(inds), :]
                    else:
                        return_data[col] = node[np.array(inds)]

        # convert data to correct format. filehandle doesn't need to be open, so do here to get data off disk first
        for col in return_data.keys():
            dtype = col_dtype[col]

            if re.match(r'o', dtype):
                return_data[col] = [msgpack_loads(x, compress=False) for x in return_data[col]]
            elif re.match(r'c', dtype):
                return_data[col] = [msgpack_loads(x, compress=True) for x in return_data[col]]
            else:
                if re.match(r's', dtype):
                    return_data[col] = np.core.defchararray.decode(return_data[col], 'utf-8')
                else:
                    return_data[col] = return_data[col]

            if flavor and flavor == 'numpy':
                pass
            elif flavor and flavor == 'python' and isinstance(return_data[col], np.ndarray):
                return_data[col] = return_data[col].tolist()
            elif ATTR_COLFLAV in simpah5_attrs and \
                    simpah5_attrs[ATTR_COLFLAV][col] == 'python' and isinstance(return_data[col], np.ndarray):
                return_data[col] = return_data[col].tolist()
            elif ATTR_COLFLAV not in simpah5_attrs and isinstance(return_data[col], np.ndarray):
                # default to python
                return_data[col] = return_data[col].tolist()

        return return_data

    def iter_column(self, table_name: str, col: str, flavor: str=''):
        """
        Iterate through all values of a single column in table.

        :param table_name:
        :param col:
        :return:
        """

        with self.open(mode='r') as h5:

            simpah5_attrs = self._table_info(h5, table_name)
            if len(simpah5_attrs) == 0:
                raise Exception(f"{table_name} doesn't exist in {self._h5file}")
            col_dtype = simpah5_attrs[ATTR_COLDTYPE]
            nrows = simpah5_attrs[ATTR_NROWS]
            return_cols = set(col_dtype.keys())
            if col not in return_cols:
                raise Exception(f'Specified column {col} not in {table_name} columns {list(return_cols)}')

            dtype = col_dtype[col]
            colflavor = simpah5_attrs[ATTR_COLFLAV][col]
            colnode = self._get_col(h5, self._path(table_name, col))
            for row in colnode.iterrows():

                if dtype[0] == 's':
                    yield row.decode('utf-8')
                elif dtype[0] == 'o':
                    yield msgpack_loads(row, compress=False)
                elif dtype[0] == 'c':
                    yield msgpack_loads(row, compress=True)
                elif flavor and flavor == 'numpy':
                    yield row
                elif dtype[0] == 'f' or dtype[0] == 'n' and colflavor == 'python':
                    yield float(row)
                elif dtype[0] == 'i' and colflavor == 'python':
                    yield int(row)
                else:
                    yield row

    def _append_ctable(self, h5, table_path: str, col_data: dict, resize: bool=True,
                       col_dtypes: Optional[dict]=None):

        if col_dtypes is None:
            col_dtypes = {}

        # ensure all lengths are equal error check
        data_lengths = [len(d) for d in col_data.values()]
        last_len = data_lengths[0]
        for klen in data_lengths:
            if last_len != klen:
                raise Exception(f"column lengths are not equal on append: {last_len} vs {klen}")

        simpah5_attrs = self._read_attrs(h5, table_path)
        # need to create table as it does not exist
        if len(simpah5_attrs) == 0:
            if col_dtypes:
                # compute shapes
                shapes = {}
                for col, data in col_data.items():
                    sp = (0,)
                    if isinstance(data, np.ndarray) and len(data.shape) > 1:
                        sp = list(data.shape)
                        sp[0] = 0
                    shapes[col] = sp

                self._create_ctable(h5, table_path, col_dtypes, shapes)
                simpah5_attrs = self._read_attrs(h5, table_path)
            else:
                coldt = {}
                for col, data in col_data.items():
                    dtype = self._create_column_from_data(h5, table_path, col, data)
                    coldt[col] = dtype
                # write table attributes
                self._write_attrs(h5, table_path, ATTR_COLDTYPE, coldt)
                _ = self._set_flavor(h5, table_path, col_data, coldt)
                return

        coldt = simpah5_attrs[ATTR_COLDTYPE]

        if len(set(coldt.keys()).difference(set(col_data.keys()))) > 0:
            raise Exception(f"Not all columns specified in col_data for append to {table_path}. "
                            f"dtypes={set(coldt.keys())} ... col_data={set(col_data.keys())} in {self._h5file}")

        # check and set the column flavor once
        if ATTR_COLFLAV not in simpah5_attrs:
            simpah5_attrs[ATTR_COLFLAV] = self._set_flavor(h5, table_path, col_data, coldt)

        prev_nrows = self._table_info(h5, table_path)[ATTR_NROWS]
        restore = False
        error_msg = ''
        for col, coldtype in coldt.items():

            try:
                data = self._convert_data(col_data[col], coldtype)
                colpath = self._path(table_path, col)

                if re.match(r'[osc](\d+)', coldtype):
                    colnode = self._safe_col_str_change(h5, colpath, coldtype, data, resize)
                else:
                    colnode = self._get_node(h5, colpath)
                    if colnode is None:
                        raise Exception(f"Table column doesn't exist: {colpath} in {self._h5file}")

                colnode.append(data)
            except:
                restore = True
                error_msg = traceback.format_exc()
                break

        if restore:
            # loop through all column nodes and ensure they have the same previous length to restore
            for col, coldtype in coldt.items():
                col_path = self._path(table_path, col)
                node = self._get_col(h5, col_path)
                cur_rows = int(node.shape[0])
                if cur_rows > prev_nrows:
                    inds = list(range(prev_nrows, cur_rows))
                    node = self._get_col(h5, col_path)
                    self._remove_array_rows(node, inds)

            raise Exception(error_msg)

    @staticmethod
    def _convert_data(data: (list, tuple), coldtype: str):

        if re.match(r'c', coldtype):
            data = [msgpack_dumps(x, compress=True) for x in data]
        elif re.match(r'o', coldtype):
            data = [msgpack_dumps(x, compress=False) for x in data]

        data = np.array(data)
        if re.search(r'U', str(data.dtype)):
            data = np.core.defchararray.encode(data, 'utf-8')

        assert isinstance(data, (list, tuple, np.ndarray))

        return data

    def _path(self, colpath: str, colname: str) -> str:

        return self._norm_path(colpath, colname=colname)

    def _get_col(self, h5: tb.File, nodepath: str) -> tb.EArray:

        nodepath = self._norm_path(nodepath)
        try:
            node = h5.get_node(nodepath)
        except KeyError:
            node = None
        return node

    def _get_node(self, h5: tb.File, nodepath: str) -> tb.Leaf:

        nodepath = self._norm_path(nodepath)
        try:
            node = h5.get_node(nodepath)
        except KeyError:
            node = None
        return node

    @staticmethod
    def _norm_path(colpath: str, colname: str = '') -> str:

        if not colpath:
            colpath = '/'
        if colpath == '/':
            name = []
        else:
            name = ['']
        cst = 0
        cend = len(colpath)
        if colpath[0] == '/':
            cst = 1
        if colpath[-1] == '/':
            cend = cend - 1
        name.append(colpath[cst:cend])

        if colname:
            nst = 0
            nend = len(colname)
            if colname[0] == '/':
                nst = 1
            if colname[-1] == '/':
                nend = nend - 1
            name.append(colname[nst:nend])

        return '/'.join(name)

    def _exists_node(self, h5: tb.File, nodepath: str) -> tb.Leaf:

        nodepath = self._norm_path(nodepath)
        try:
            node = h5.get_node(nodepath)
        except tb.NoSuchNodeError:
            node = None
        return node

    def _write_attrs(self, h5, table_path, attrs_name, attrs_value):
        table = self._get_node(h5, table_path)
        try:
            attr_bytes = table._v_attrs[ATTR_CTABLE]
            simpah5_attrs = msgpack_loads(attr_bytes, use_list=True)
        except KeyError:
            simpah5_attrs = {}

        simpah5_attrs['_version'] = VERSION
        simpah5_attrs[attrs_name] = attrs_value
        table._v_attrs[ATTR_CTABLE] = msgpack_dumps(simpah5_attrs)

    def _read_attrs(self, h5, table_path: str) -> dict:
        try:
            node = self._get_node(h5, table_path)
        except tb.NoSuchNodeError:
            return {}
        if ATTR_CTABLE not in node._v_attrs:
            return {}
        simpah5_attrs = msgpack_loads(node._v_attrs[ATTR_CTABLE], use_list=True)
        return simpah5_attrs

    def _set_flavor(self, h5: tb.File, table_path: str, col_data: dict, col_dtypes: dict) -> dict:

        # set column flavors
        colflavor = {}
        for col, dtype in col_dtypes.items():
            if isinstance(col_data[col], (list, tuple)):
                colflavor[col] = 'python'
            else:
                colflavor[col] = 'numpy'
        self._write_attrs(h5, table_path, ATTR_COLFLAV, colflavor)

        return colflavor

    def _create_column(self, h5: tb.File, colpath: str, atom: Optional[tb.Atom]=None, expectedrows: int=10000,
                       shape: Optional[tuple]=None, data: (list, tuple, np.ndarray)=None) -> tb.EArray:
        # create an EArray column and return the created node

        if data is None and shape is None:
            shape = (0,)

        if data is not None and not isinstance(data, np.ndarray) and isinstance(data[0], str):
            data = [x.encode('utf-8') for x in data]

        return h5.create_earray(
            os.path.dirname(colpath), os.path.basename(colpath), obj=data, createparents=True,
            atom=atom, shape=shape, expectedrows=expectedrows, filters=self._filters
        )

    def _create_column_from_dtype(self, h5: tb.File, table_path: str, col_name: str, col_dtype: str, shape: tuple,
                                  expectedrows: int=10000):

        colpath = self._path(table_path, col_name)

        if re.match(r'[nf]', col_dtype):
            self._create_column(h5, colpath, atom=tb.Float64Atom(), shape=shape, expectedrows=expectedrows)

        elif re.match(r'i', col_dtype):
            self._create_column(h5, colpath, atom=tb.Int64Atom(), shape=shape, expectedrows=expectedrows)

        elif re.match(r'[osc](\d+)', col_dtype):
            m = re.match(r'[osc](\d+)', col_dtype)
            size = int(m.group(1))
            self._create_column(h5, colpath, atom=tb.StringAtom(size), shape=shape, expectedrows=expectedrows)

        else:
            raise Exception(f'Unrecognized col_dtype: {col_dtype}')

    def _create_column_from_data(self, h5: tb.File, table_path: str, col_name: str, data: (list, tuple, np.ndarray),
                                 expectedrows: int=10000) -> str:

        if isinstance(data[0], str):
            objdt = 's'

        elif isinstance(data[0], (list, tuple, dict, bytes)):
            data = [msgpack_dumps(x, compress=False) for x in data]
            objdt = 'o'

        elif not isinstance(data[0], (int, float, np.int, np.float, np.ndarray)):
            raise Exception(f"Unknown type in col: {col_name} type:{type(data[0])} in {self._h5file}")

        earray_col = self._create_column(h5, self._path(table_path, col_name), data=data, expectedrows=expectedrows)

        if re.match(r'[if]', str(earray_col.dtype)):
            m = re.match(r'([if])', str(earray_col.dtype))
            dtype = m.group(1)

        elif re.search(r'S(\d+)', str(earray_col.dtype)):
            m = re.search(r'S(\d+)', str(earray_col.dtype))
            dtype = f'{objdt}{m.group(1)}'

        else:
            raise Exception(f"Corruption possible: Unallowed datatype in created column "
                            f"{table_path}/{col_name}: {str(earray_col.dtype)} in {self._h5file}")

        return dtype

    def delete_rows(self, table_path: str, query: Union[list, tuple]=(), rows: Optional[list]=None) -> None:
        """
        Delete the rows in the ctable.   Specified by EITHER query OR rows.   If rows is specified those rows will
        be deleted. IMPORTANT: rows may be shuffled in the table after delete to free space,
        so row order will not be preserved.

        For query use see :meth:`.read_ctable`

        :param table_path: path to table.
        :param query: rows meeting these conditions will be deleted.
        :param rows: list of integers corresponding to the rows in the table that should be deleted.
        :return:
        """

        if query and rows:
            raise Exception(f"Both query and rows specified in {self._h5file}")
        elif not query and not rows:
            raise Exception(f"Either query/match or rows MUST be specified in {self._h5file}")

        with self.open(mode='a') as h5:

            simpah5_attrs = self._read_attrs(h5, table_path)
            col_dtype = simpah5_attrs[ATTR_COLDTYPE]

            if rows:
                inds = rows

            elif query:
                # return a dictionary of node name to node pointers
                node_data = h5.get_node(self._norm_path(table_path))._v_children
                inds = _filter_inds(node_data, query)
                inds = np.nonzero(inds)[0]

            else:
                raise NotImplementedError('match not implemented yet')

            if len(inds) != 0:
                for col in col_dtype:
                    col_path = self._path(table_path, col)
                    node = self._get_col(h5, col_path)
                    self._remove_array_rows(node, inds)

    def _remove_array_rows(self, node: tb.EArray, rm_rows: Union[list, tuple]):
        """
        Reorganize array by re-writing(moving) the minimal number of rows in order to delete all specified
        rows and to truncate the array without losing data.

        * any rows after len(node)-len(rm_rows) must be moved since they will be truncated
        * only those rows after len(node)-len(rm_rows) that are to be kept must be moved
          ensure same rows are not overwritten

        :param node: h5 node
        :param rm_rows: list of row indicies to remove
        :return:
        """

        node_len = len(node)

        rm_rows = np.array(sorted(set(rm_rows)))
        num_rm = len(rm_rows)

        # determine last row index (non-inclusive) that needs to be moved
        last_row = node_len - num_rm
        # compute the set of rows that are past the last row
        dont_worry_rows = set(rm_rows[rm_rows >= last_row])
        # compute the rows that need to be moved
        move_rows = set(range(last_row, node_len)).difference(dont_worry_rows)
        move_rows = sorted(move_rows)
        # compute the new set of rows to replace with the move rows
        replace_rows = sorted(set(rm_rows[rm_rows < last_row]))

        if len(move_rows) != len(replace_rows):
            raise Exception(f'Something is wrong with the length of the move {len(move_rows)} and replace '
                            f'{len(replace_rows)} rows in {self._h5file}')
        # shift rows at the end of the file into the row to remove to minimize number of writes
        for i in range(len(move_rows)):
            node[replace_rows[i]] = node[move_rows[i]]
        # delete the end of the array that was moved to the removed nodes
        node.truncate(len(node) - num_rm)

    def table_info(self, table_name: str) -> dict:
        """
        Read the table information including column datatypes and flavor.  Returns empty dictionary if table doesn't
        exist.  Info returned as keys of dictionary::

            * col_dtype: dict of column data types
            * col_flavor: dict of column flavors
            * num_rows: number of rows in table

        :param table_name: table name

        :return
        """

        with self.open(mode='r') as h5:
            return self._table_info(h5, table_name)

    def table_nrows(self, table_name: str) -> dict:
        """
        Return the number of rows in the table
        :param table_name:
        :return:
        """
        return self.table_info(table_name).get(ATTR_NROWS, 0)

    def _table_info(self, h5, table_name: str) -> dict:

        simpah5_attrs = self._read_attrs(h5, table_name)
        if len(simpah5_attrs) > 0:
            for colname in simpah5_attrs[ATTR_COLDTYPE]:
                node = self._get_col(h5, self._path(table_name, colname))
                simpah5_attrs[ATTR_NROWS] = int(node.shape[0])
                break

        return simpah5_attrs

    def _safe_col_str_change(self, h5: tb.File, colpath: str, coldtype: str, data: (list, tuple), resize: bool):

        colnode = self._get_col(h5, colpath)
        if colnode is None:
            raise Exception(f"Table column doesn't exist: {colpath} in {self._h5file}")

        m = re.match(r'[osc](\d+)', coldtype)
        if not m:
            raise Exception(f'Col dtype for column {colpath} should be [osc] and is not: {coldtype} in {self._h5file}')
        size = int(m.group(1))

        m = re.search(r's(\d+)', str(data.dtype).lower())
        if not m:
            raise Exception(f'Data in column {colpath} should be similar to {coldtype} and is {data.dtype}'
                            f' in {self._h5file}')
        dlen = int(m.group(1))
        if dlen > size:
            if not resize:
                msg = f"Data corruption happening in {colpath}. Table may be corrupted." \
                      f"Serialized data len ({dlen}) > ({size + 1}) in {self._h5file}"
                raise Exception(msg)
            else:

                logging.warning(f"Changing column size to {dlen} and overwriting ... {colpath}")
                # safely rewrite data at expense of memory and time

                tmp_col_path = colpath + '_tmp'
                # create new column of desired shape
                newcolnode = self._create_column(h5, tmp_col_path, atom=tb.StringAtom(dlen))
                # re-write all data into that column
                # newcolnode.append(colnode[:])
                for idx, row in enumerate(colnode.iterrows()):
                    newcolnode.append([row])

                # clean up old path and point to new path
                h5.remove_node(colpath)
                colname = os.path.basename(colpath)
                table_path = os.path.dirname(colpath)
                h5.rename_node(tmp_col_path, colname)

                # update attributes
                colnode = self._get_col(h5, colpath)
                m = re.search(r'S(\d+)', str(colnode.dtype))
                new_len = int(m.group(1))
                simpah5_attrs = self._read_attrs(h5, table_path)
                col_dtypes = simpah5_attrs[ATTR_COLDTYPE]
                m = re.match(r'([os])', col_dtypes[colname])
                col_dtypes[colname] = f'{m.group(1)}{new_len}'
                _ = self._write_attrs(h5, table_path, ATTR_COLDTYPE, col_dtypes)

        return colnode

    def repack(self) -> None:
        """
        Re-write entire file which has the effect of recompressing data efficiently and eliminating free space.

        Should generally be run periodically if files are written to largely or modified.

        :return:
        """

        tmp_name = str(uuid.uuid4()) + '.' + self._h5file
        with self.open(mode='r') as h5:
            h5.copy_file(tmp_name, filters=self._filters)
        os.remove(self._h5file)
        os.rename(tmp_name, self._h5file)
        # write repack timestamp
        with self.open(mode='a') as h5:
            data = [datetime.now().timestamp(), h5.get_filesize()]
            try:
                h5.get_node('/_last_repack')[:] = data
            except AttributeError:
                h5.create_array('/', '_last_repack', obj=data)


