"""
Some characterization of the speed of h5 data access
"""

import numpy as np
import random
import time
import os
import tables as tb

from pyh5column import H5ColStore


def test2(complib='blosc'):
    num_rows = 10000
    max_len = 100
    c = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    l = len(c)
    # write a single column of random strings to carray

    # generate a lists of random strings
    min_len = int(0.5*max_len)  # fixed length if min_len=max_len, otherwise variable length strings
    a = [[c[i] for i in np.random.random_integers(0, l-1, np.random.randint(min_len, max_len+1, 1))]
         for _ in range(num_rows)]
    sa = ["".join(b) for b in a]
    tot_char = np.sum([len(a[i]) for i in range(len(a))])
    npar = np.array(sa)

    fname = 'test.h5'
    remove_file(fname)

    sttime = time.time()
    h5 = SimpaH5('test.h5', complib=complib, complevel=9)
    h5.append_strs(npar, '/test_array')
    wtime = time.time() - sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("write String array: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" %
          (char_p_sec, rows_p_sec, wtime))

    sttime = time.time()
    r = h5.read_node('/test_array')
    wtime = time.time() - sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("read Stirng array: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" %
          (char_p_sec, rows_p_sec, wtime))
    remove_file(fname)


def test1(complib=b'blosc'):
    num_rows = 10000
    max_len = 100
    c = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    l = len(c)
    # write a single column of random strings to carray

    # generate a lists of random strings
    min_len = int(0.5*max_len)  # fixed length if min_len=max_len, otherwise variable length strings
    a = [[c[i] for i in np.random.random_integers(0, l-1, np.random.randint(min_len, max_len+1, 1))]
         for _ in range(num_rows)]
    sa = ["".join(b) for b in a]
    tot_char = np.sum([len(a[i]) for i in range(len(a))])
    npar = np.array(sa)

    fname = 'test.h5'
    remove_file(fname)

    h5 = SimpaH5('test.h5', complib=complib, complevel=9)
    sttime = time.time()
    h5.write_carray(npar, 'test_array')
    wtime = time.time()-sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("write CArray: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" % (char_p_sec, rows_p_sec, wtime))

    sttime = time.time()
    r = h5.read_node('/test_array')
    wtime = time.time()-sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("read CArray: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" % (char_p_sec, rows_p_sec, wtime))
    remove_file(fname)

    # compute total characters written
    sttime = time.time()
    h5 = SimpaH5('test.h5', complib=complib, complevel=9)
    h5.write_vlarray(a, 'test_array')
    wtime = time.time()-sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("write VLArray: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" % (char_p_sec, rows_p_sec, wtime))

    sttime = time.time()
    r = h5.read_node('/test_array')
    wtime = time.time()-sttime
    rows_p_sec = num_rows / wtime
    char_p_sec = tot_char / wtime
    print("read VLArray: char_p_sec = %.2e, rows per sec = %.2e, total=%.3f sec" % (char_p_sec, rows_p_sec, wtime))
    remove_file(fname)


def check_many_cols_simpa(n=200):
    loops = 5

    print("\n****Simpa HDF5 TEST")
    fname = 'testmany.h5'
    remove_file(fname)

    h5 = H5ColStore(fname)

    col_dtype = {f'x{x}': 'n' for x in range(n)}
    col_dtype['date'] = 's10'

    # create data
    col_data = {}
    for x in col_dtype:
        col_data[x] = [random.random() for i in range(100)]
    col_data['date'] = [f'{x}' for x in range(100)]

    for i in range(loops):
        sttime = time.time()
        h5.append_ctable('/data', col_data, col_dtypes=col_dtype)
        print("Write time:", time.time()-sttime)

    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data')
        print("ALL read", time.time()-sttime)

    # read specific row slice
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', query=[['date', [('==', '1')]]])
        print(f"Read single row {len(d)} {len(d['x1'])}", time.time()-sttime)

    # read column slice
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=['x1'])
        print(f"Read single column {len(d['x1'])}", time.time()-sttime)

    # read single column with match
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=['x1'], query=[['date', [('==', '1')]]])
        print(f"Read single column match {len(d['x1'])}", time.time() - sttime)

    remove_file(fname)


def check_many_cols_tables(n=200):

    loops = 5
    print("\n****TABLES TEST")
    fname = 'testmanypytb.h5'
    remove_file(fname)

    col_dtype = {f'x{x}': tb.Float64Col() for x in range(n)}
    col_dtype['date'] = tb.StringCol(10)

    h5file = tb.open_file(fname, mode="w", title="Test file")
    table = h5file.create_table(h5file.root, 'test', col_dtype)
    h5file.close()

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="a", title="Test file")
        row = h5file.get_node('/test').row
        # write data
        for i in range(100):
            for x in col_dtype:
                row[x] = random.random()
            row['date'] = f'{i}'
            row.append()
        h5file.flush()
        h5file.close()
        print("Write time:", time.time()-sttime)

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test').read()
        h5file.close()
        print(f"Read ALL: {np.shape(table)}", time.time()-sttime)

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        vals = [row.fetch_all_fields() for row in table.where('date == b"1"')]
        h5file.close()
        print(f"Read single row: {len(vals[0])} {len(vals)}", time.time()-sttime)

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        col = h5file.get_node('/test').col('x1')
        h5file.close()
        print(f"Read single column: {len(col)}", time.time()-sttime)

    # single column with match
    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        vals = [row['x1'] for row in table.where('date == b"1"')]
        h5file.close()
        print(f"Read single column match {len(vals)}", time.time() - sttime)

    remove_file(fname)


def remove_file(fname):
    try:
        os.remove(fname)
    except:
        pass


def test_blocks():

    testit = test2

    print("Lz4")
    testit(complib='blosc:lz4')
    print("BLOSC")
    testit(complib='blosc')
    print("BLOSC-LZ")
    testit(complib='blosc:blosclz')
    print("ZLIB")
    testit(complib='zlib')


if __name__ == "__main__":
    # test_blocks()
    check_many_cols_tables(n=200)
    check_many_cols_simpa(n=200)
