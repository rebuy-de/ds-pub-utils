import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pubdsutils import calcs as ca
from collections import OrderedDict


class TestValueCounts(unittest.TestCase):

    def test_value_counts_comb(self):
        s = pd.Series([1, 2, 2, 2])
        expected = pd.concat(
            [
                pd.Series([1 / 4, 3 / 4], index=pd.Int64Index([1, 2])),
                pd.Series([1, 3], index=pd.Int64Index([1, 2]))
            ],
            keys=['Ratio', 'Count'], axis=1
        ).sort_values('Ratio', ascending=False)
        assert_frame_equal(ca.value_counts_comb(s), expected)

    def test_value_counts_comb_df(self):
        df = pd.DataFrame({
            "v1": [1, 2, 2, 2],
            "v2": ['a', 'b', 'c', 'a']
        }, index=['foo', 'bar', 'hello', 'world'])
        expected = pd.concat(
            [
                pd.Series([1 / 4, 3 / 4], index=pd.Int64Index([1, 2])),
                pd.Series([1, 3], index=pd.Int64Index([1, 2]))
            ],
            keys=['Ratio', 'Count'], axis=1
        ).sort_values('Ratio', ascending=False)
        print(ca.value_counts_comb(df.v1))
        assert_frame_equal(ca.value_counts_comb(df.v1), expected)
