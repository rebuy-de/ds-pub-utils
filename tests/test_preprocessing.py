import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from pubdsutils import preprocessing as pp
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class TestColumnsOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "d1": [1, 2, 3],
                "d2": [0, 1, 1]
            }
        ), index=['foo', 'bar', 'goo'])

    def test_basic(self):
        res = pp.ColumnsOneHotEncoder(cols=['d1'], n_values=7).fit_transform(
            self.df)

        expected_res = pd.DataFrame(OrderedDict(
            {
                'd2':   [0, 1, 1],
                'd1_0': [0., 0., 0.],
                'd1_1': [1., 0., 0.],
                'd1_2': [0., 1., 0.],
                'd1_3': [0., 0., 1.],
                'd1_4': [0., 0., 0.],
                'd1_5': [0., 0., 0.],
                'd1_6': [0., 0., 0.]
            }
        ), index=['foo', 'bar', 'goo'])

        assert_frame_equal(res, expected_res)

    def test_wrong_colums(self):
        self.assertRaises(ValueError, pp.ColumnsOneHotEncoder(
            cols=['d1'], n_values=2).fit, self.df)

    def test_errors(self):
        self.assertRaises(ValueError, pp.ColumnsOneHotEncoder)
        self.assertRaises(ValueError, pp.ColumnsOneHotEncoder,
                          cols=['foo'], n_values='bar')


class TestStandartizeFloatCols(unittest.TestCase):

    def setUp(self):
        self.v1 = np.array([1., 2., 3.])
        self.v1_mean = self.v1.mean()
        self.v1_std = self.v1.std()
        self.v2 = np.array([1., 4., 8.])
        self.v2_mean = self.v2.mean()
        self.v2_std = self.v2.std()
        self.arr = np.array([self.v1, self.v2]).T
        self.df = pd.DataFrame({
            'v1': self.v1,
            'v2': self.v2
        })

    def test_both_columns(self):
        assert_array_equal(
            pp.StandartizeFloatCols(
                cols=['v1', 'v2']).fit_transform(self.df).values,
            StandardScaler().fit_transform(self.arr)
        )

    def test_both_columns_df(self):
        assert_frame_equal(
            pp.StandartizeFloatCols(cols=['v1', 'v2']).fit_transform(self.df),
            pd.DataFrame(StandardScaler().fit_transform(
                self.arr), columns=['v1', 'v2'])
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

    def test_fit_on_self_transform_on_another(self):
        """This test stress the intended behavior that the class
        can be trained on one data set X and using the fitting transfrom
        a different data set X'
        """

        sfc = pp.StandartizeFloatCols(cols=['v1', 'v2'])
        sfc.fit(self.df)

        v1 = np.array([1., 1., 1.])
        v2 = np.array([5., 5., 5.])

        new_df = pd.DataFrame(
            {
                'v1': v1,
                'v2': v2
            }
        )

        expected_res = pd.DataFrame(
            {
                'v1': (v1 - self.v1_mean) / self.v1_std,
                'v2': (v2 - self.v2_mean) / self.v2_std
            }
        )

        assert_frame_equal(
            sfc.transform(new_df),
            expected_res
        )

    def test_errors(self):
        self.assertRaises(
            NotFittedError,
            pp.StandartizeFloatCols(cols=['v1']).transform, self.df)
        self.assertRaises(ValueError, pp.StandartizeFloatCols)
        self.assertRaises(ValueError, pp.StandartizeFloatCols, cols=[1, ])
        self.assertRaises(ValueError, pp.StandartizeFloatCols(
            cols=['v3']).fit, self.df)


class TestRemoveConstantColumns(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 1, 1, 1],
                "v2": [1, 2, 3, 4],
                "v3": ['foo', 'foo', 'foo', 'foo'],
                "v4": [1.000001, 1, 1, 1]
            }
        ))

    def test_defaults(self):
        assert_frame_equal(
            pp.RemoveConstantColumns().fit_transform(self.df),
            pd.DataFrame(OrderedDict(
                {
                    "v2": [1, 2, 3, 4],
                    "v4": [1.000001, 1, 1, 1]
                }
            ))
        )

    def test_const_cols_attr(self):
        rcc = pp.RemoveConstantColumns()
        rcc.fit(self.df)
        self.assertListEqual(
            rcc.const_cols.tolist(),
            ["v1", "v3"]
        )
