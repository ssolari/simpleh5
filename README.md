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

Why
===

The HDF5 format along with PyTables enables the ability to store/access data very efficiently on disk in a single
file.  Data can be compressed and organized in a hierarchical manner.   PyTables is an excellent library for
writing and reading HDF5, however it only provides the ability to store tables as a collection of rows.  Further, there
is a lot of boilerplate that needs to be handled.  In many
analytic usecases the storage of a table as a collection of columns can be more efficient in compression and
performance.   Reading a column from simpleh5 H5ColStore (using pytables) is 10x-20x faster than reading a column 
out of a pyTables Table.   Programer convenience is also essential.  Adding and removing columns at will is useful 
in feature engineering.  Storing data in rows makes this difficult and slow.  SimpaH5 attempts to abstract the 
storage of tables as collections of columns of data.

Performance is also a factor of network bandwidth.   The transfer of data across that bandwidth matters.  The 10x-20x 
gain in column read speed is on a local drive.  When a network or cloud based drive is read the performance can be even
larger.  When a pytables Table is read, every row needs to be transfered to the CPU across the network.  When a column
is read only the data in that column needs to be transfered.   This helps with complex analytic queries across the data
involving selecting a set of rows based on information in a small set of columns.  In this case, only the columns in the
query are read, eliminating the need to index data.

SimpleH5 also allows the seemless storage of any python objects serializable with msgpack, with the addition of numpy structures.   
In this way
arbitrary objects can be saved efficiently.   Compression can also be applied to the objects to limit the needed size
for a given type definition of a column.  While this is double compression may seem strange, the effect is smaller final
file sizes. 

Together certain 'tricks' emerge to store real-world data with far more ease.  For example, take a string column where some strings
are large (say 1MB) and most strings are small (say 1kB) and there are 10s of millions of rows.   Typical fixed storage might
require 1e6 bytes (column size) x 10e6 rows ~ 10TB size of column.   However, HDF5 compression means that the column is compressed
and the 1kB strings do not take up the full 1MB, so the column size in practice may be less than 1GB.   If we define the column
as a compressed object (in effect storing compressed strings inside a compressed column) then the size of the column may only be
200kB (i.e. the size of the longest compressed string) and the overall column size will be even smaller.


Simpleh5 adds a few extra conveniences for 'real-life' use cases.   One such convenience is that while HDF5 columns need
to have a fixed data size, that size does not need to be specified a-priori when writing data with SimpleH5.   If SimpleH5 
encounters data in a column larger than the size of that column it automatically expands the column 
(this default behavior can be turned off).  The mechanism effectively copies the column to a new column with larger size, 
then appends the new data to this column and deletes the old.  For this convenience, two tradeoffs exist:

  1) Speed of writes.  Every time a column is expanded costs extra time during that individual write, which is a function
  of the current size of the column.  So writing the largest data first, means that all writes after are fast. 
  2) Size of file.  HDF5 does not easily free memory, so the filesize may increase after a column is expanded.  Using the
  repack function on a file re-writes the file to minimum size and optimal organization.
  
A benefit of storing compressed objects is also that the impact of expanding columns can be less in many cases, such as
the compressed string example above.

Another convenience is that Simpleh5 will read the datatypes of the first data stored into a table and use those datatypes, 
however specifying the column datatypes is more appropriate to ensure data is written as expected.

Cautions
========

One of the biggest issues of using SimpleH5 is that storing data one row at a time can cause the file size to grow
quickly, because of a lack of efficient compression.  Data is stored in 'blocks' of columns, therefore, storing single
elements in a column one at a time does not efficiently compress the blocks.   Writing a large number of rows in a
single append is therefore more efficient.

The solution is to periodically perform
a .repack() on the file, which essentially re-copies and then re-names the entire file.   The result is a file
that can be upto 2 orders of magnitude smaller than the original file (depending how much data was stored and how it
was stored).

