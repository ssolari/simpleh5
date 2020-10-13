"""
Test issues related to msgpack
Note: msgpack  encodes to binary which may include needed null bytes b'\x00' at end.  Therefore caution is needed
when pushing an encoded binary string with end null bytes to numpy, because numpy will truncate all trailing null bytes.
"""

import unittest
import msgpack
import numpy as np
from pyh5column.utilities.serialize_utilities import str_dtype, obj_dtype

import sys
PY_VER = sys.version[0]


class TestMsgPack(unittest.TestCase):

    def test_np_msgpack_float(self):
        s = np.array([1, 2, 1, 1]).astype(np.float)
        r = msgpack.unpackb(msgpack.packb(list(s), use_bin_type=True, use_single_float=False), raw=False)
        self.assertListEqual(list(s), r)

    def test_np_msgpack_int(self):
        s = [1, 2, 1, 1]
        r = msgpack.unpackb(msgpack.packb(list(s), use_bin_type=True, use_single_float=False), raw=False)
        self.assertListEqual(list(s), r)
        # fails only in python 3 due to numpy integer support of msgpack
        if PY_VER == '3':
            with self.assertRaises(TypeError):
                msgpack.dumps(list(np.array(s)))
        else:
            self.assertListEqual(list(s), r)


class TestUtils(unittest.TestCase):

    def test_longests_str(self):
        s = ['abcdefg', 'abcdefghij£']
        self.assertEqual('s12', str_dtype(s))

    def test_longests_obj(self):
        s = [['abcdefg', 'abcdefghij£'], ['a', 'b', 'c']]
        self.assertEqual('o39', obj_dtype(s))


if __name__ == "__main__":
    unittest.main()
