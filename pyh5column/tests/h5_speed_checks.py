"""
Some characterization of the speed of h5 data access, comparing PyTables Table access to data vs accessing
via columnar store in pyh5column.

"""


import numpy as np
import random
import time
import os
import tables as tb

from pyh5column import H5ColStore


loops = 5
num_columns = 200
num_rows = 20000  # added each loop, total rows written will be loops*num_rows

WA = 'Write all'
RA = 'Read all'
RSR = 'Read single row'
RSC = 'Read single column'
R10C = 'Read 10 columns'
RSCM = 'Read single column match few rows'
R10P = 'Read 10 columns with ~10% search from float range'


def check_many_cols_raw_tables():

    time_averages = {
        WA: [],
        RA: [],
        RSR: [],
        RSC: [],
        R10C: [],
        RSCM: [],
        R10P: [],
    }
    print("\n****TABLES TEST")
    fname = 'testfile_speed_check_tables.h5'
    remove_file(fname)

    col_dtype = {f'x{x}': tb.Float64Col() for x in range(num_columns)}
    col_dtype['date'] = tb.StringCol(10)

    h5file = tb.open_file(fname, mode="w", title="Test file")
    table = h5file.create_table(h5file.root, 'test', col_dtype)
    h5file.close()

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="a", title="Test file")
        row = h5file.get_node('/test').row
        # write data
        for i in range(num_rows):
            for x in col_dtype:
                row[x] = random.random()
            row['date'] = f'{i}'
            row.append()
        h5file.flush()
        h5file.close()
        time_averages[WA].append(time.time()-sttime)
    print(f"  Avg {WA}: {np.mean(time_averages[WA])}")

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test').read()
        h5file.close()
        time_averages[RA].append(time.time()-sttime)
    print(f"  Avg {RA}: {np.mean(time_averages[RA])}")

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        vals = [row.fetch_all_fields() for row in table.where('date == b"1"')]
        h5file.close()
        time_averages[RSR].append(time.time()-sttime)
    print(f"  Avg {RSR}: {np.mean(time_averages[RSR])}")

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        col = h5file.get_node('/test').col('x1')
        h5file.close()
        time_averages[RSC].append(time.time()-sttime)
    print(f"  Avg {RSC}: {np.mean(time_averages[RSC])}")

    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        cols = {f'x{n}': [] for n in range(10)}
        for row in table.iterrows():
            for n in range(10):
                cols[f'x{n}'].append(row[f'x{n}'])
        h5file.close()
        time_averages[R10C].append(time.time()-sttime)
    print(f"  Avg {R10C}: {np.mean(time_averages[R10C])}")

    # single column with match
    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        vals = [row['x1'] for row in table.where('date == b"1"')]
        h5file.close()
        time_averages[RSCM].append(time.time()-sttime)
    print(f"  Avg {RSCM}: {np.mean(time_averages[RSCM])}")

    # read 10 columns with 10% float match like a range
    for j in range(loops):
        sttime = time.time()
        h5file = tb.open_file(fname, mode="r", title="Test file")
        table = h5file.get_node('/test')
        cols = {f'x{n}': [] for n in range(10)}
        for row in table.where('(x1 > 0.1) & (x1 < 0.2)'):
            for n in range(10):
                cols[f'x{n}'].append(row[f'x{n}'])
        h5file.close()
        time_averages[R10P].append(time.time() - sttime)
    print(f"  Avg {R10P}: {np.mean(time_averages[R10P])}")

    remove_file(fname)

    return time_averages


def check_many_cols_pyh5col():

    time_averages = {
        WA: [],
        RA: [],
        RSR: [],
        RSC: [],
        R10C: [],
        RSCM: [],
        R10P: [],
    }
    print("\n****PH5Column HDF5 TEST")
    fname = 'testfile_speed_check_pyh5col.h5'
    remove_file(fname)

    h5 = H5ColStore(fname)

    col_dtype = {f'x{x}': 'n' for x in range(num_columns)}
    col_dtype['date'] = 's10'

    # create data
    col_data = {}
    for x in col_dtype:
        col_data[x] = [random.random() for i in range(num_rows)]
    col_data['date'] = [f'{x}' for x in range(num_rows)]

    for i in range(loops):
        sttime = time.time()
        h5.append_ctable('/data', col_data, col_dtypes=col_dtype)
        time_averages[WA].append(time.time()-sttime)
    print(f"  Avg {WA}: {np.mean(time_averages[WA])}")

    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data')
        time_averages[RA].append(time.time()-sttime)
    print(f"  Avg {RA}: {np.mean(time_averages[RA])}")

    # read specific row slice
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', query=['date', '==', '1'])
        time_averages[RSR].append(time.time()-sttime)
    print(f"  Avg {RSR}: {np.mean(time_averages[RSR])}")

    # read column slice
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=['x1'])
        time_averages[RSC].append(time.time()-sttime)
    print(f"  Avg {RSC}: {np.mean(time_averages[RSC])}")

    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=[f'x{n}' for n in range(10)])
        time_averages[R10C].append(time.time()-sttime)
    print(f"  Avg {R10C}: {np.mean(time_averages[R10C])}")

    # read single column with match
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=['x1'], query=['date', '==', '1'])
        time_averages[RSCM].append(time.time()-sttime)
    print(f"  Avg {RSCM}: {np.mean(time_averages[RSCM])}")

    # read 10 columns with 10% float range match
    for i in range(loops):
        sttime = time.time()
        d = h5.read_ctable('/data', cols=[f'x{n}' for n in range(10)], query=[('x1', '>', 0.1), ('x1', '<', 0.2)])
        time_averages[R10P].append(time.time() - sttime)
    print(f"  Avg {R10P}: {np.mean(time_averages[R10P])}")

    remove_file(fname)

    return time_averages


def remove_file(fname):
    try:
        os.remove(fname)
    except:
        pass


if __name__ == "__main__":

    print('Creating table with parameters:')
    print(f' {num_columns} columns')
    print(f' {num_rows*loops} rows (written in {num_rows} chunks)')
    print(f' {loops} loops for average timing')

    table_times = check_many_cols_raw_tables()
    col_times = check_many_cols_pyh5col()

    print("\n****Comparison")
    for k in table_times:
        print(f'\n{k} average:')
        print(f'  Tables: {np.mean(table_times[k])}')
        print(f'  PyH5Col: {np.mean(col_times[k])}')
        print(f'  Tables/PyH5Col ratio: {np.mean(table_times[k])/np.mean(col_times[k])}')


