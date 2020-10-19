# simpleh5

A hdf5 table column store built on top of pytables

H5ColStore is the top-level object to use to write and read tabular column stores.   The H5ColStore object is passed a string
representing the full file path to the desired .h5(hdf5) file.   Instantiating the object is harmless, all future operations
will operate on this .h5 file.

Example::

    my_column_store = H5ColStore('all_tables.h5')

The goal of H5ColStore is to create a simplified experience with high performance for data scientists to focus on
analytics at the trade off of 'optimal' performance.   Time and effort are a trade off and H5ColStore tries to enable
a focus on using data in an efficient manner rather than providing hooks into all parameterization of how a file can
be stored and what compressors are used.
