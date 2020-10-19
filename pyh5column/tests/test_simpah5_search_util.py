

import unittest
import os

from simpleh5.utilities.search_utilities import _build_search_string


class TestBuildString(unittest.TestCase):

    def test_string_single(self):
        query = ['strs', '==', 'abc']
        match_string, uservars = _build_search_string(query)
        self.assertEqual(match_string, "(n0==b'abc')")
        self.assertDictEqual(uservars, {'n0': 'strs'})

    def test_string_single_unicode(self):
        query = (
            ('strs', '==', '£')
        )
        match_string, uservars = _build_search_string(query)
        self.assertEqual(match_string, "(n0==b'\\xc2\\xa3')")
        self.assertDictEqual(uservars, {'n0': 'strs'})

    def test_string_compound(self):

        query = [
            [('strs', '==', 'abc'), ('strs', '==', '£')],
            ['nums', '>', 1.3]
        ]

        match_string, uservars = _build_search_string(query)
        self.assertEqual(match_string, "((n0==b'abc')|(n0==b'\\xc2\\xa3'))&(n1>1.3)")
        self.assertDictEqual(uservars, {'n0': 'strs', 'n1': 'nums'})

    def test_string_double_compound(self):
        # meaningless logic
        query = [
            [('strs', '!=', 'abc'), ('strs', '!=', 'cba')],
            ('strs', '==', '£')
        ]
        match_string, uservars = _build_search_string(query)
        self.assertEqual(match_string, "((n0!=b'abc')|(n0!=b'cba'))&(n0==b'\\xc2\\xa3')")
        self.assertDictEqual(uservars, {'n0': 'strs'})


if __name__ == "__main__":
    unittest.main()
