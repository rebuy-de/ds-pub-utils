import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from pubdsutils import preprocessing as pp
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler


class TestColumnsOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "d1": [1,2,3],
                "d2": [0,1,1]
            }
        ))

    def test_basic(self):
        res = pp.ColumnsOneHotEncoder(cols=['d1'], n_values=7).fit_transform(
            self.df)

        expected_res = pd.DataFrame(OrderedDict(
            {
                'd2':   [0,1,1],
                'd1_0': [0.,0.,0.],
                'd1_1': [1.,0.,0.],
                'd1_2': [0.,1.,0.],
                'd1_3': [0.,0.,1.],
                'd1_4': [0.,0.,0.],
                'd1_5': [0.,0.,0.],
                'd1_6': [0.,0.,0.]
            }
        ))

        assert_frame_equal(res, expected_res)

    def test_wrong_colums(self):
        self.assertRaises(ValueError, pp.ColumnsOneHotEncoder(cols=['d1'], n_values=2).fit, self.df)


class TestStandartizeFloatCols(unittest.TestCase):

    def setUp(self):
        self.v1 = np.array([1., 2., 3.])
        self.v2 = np.array([1., 4., 8.])
        self.arr = np.array([self.v1,self.v2]).T
        self.df = pd.DataFrame({
            'v1': self.v1,
            'v2': self.v2
        })

    def test_both_columns(self):
        assert_array_equal(
            pp.StandartizeFloatCols(cols=['v1', 'v2']).fit_transform(self.df).values,
            StandardScaler().fit_transform(self.arr)
        )

    def test_both_columns_df(self):
        assert_frame_equal(
            pp.StandartizeFloatCols(cols=['v1', 'v2']).fit_transform(self.df),
            pd.DataFrame(StandardScaler().fit_transform(self.arr), columns=['v1', 'v2'])
        )

    def test_one_col(self):
        assert_array_equal(
            pp.StandartizeFloatCols(
                cols=['v1']).fit_transform(self.df).values,
            np.array(
                [
                    self.v2,
                    StandardScaler().fit_transform(
                        self.v1.reshape(-1, 1)).reshape(1, -1)[0]
                ]
            ).T
        )

    def test_errors(self):
        self.assertRaises(ValueError, pp.StandartizeFloatCols)
        self.assertRaises(ValueError, pp.StandartizeFloatCols,cols=[1,])
        self.assertRaises(ValueError, pp.StandartizeFloatCols(cols=['v3']).fit, self.df)
