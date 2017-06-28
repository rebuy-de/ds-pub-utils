import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pubdsutils import data_preparation as dp
from collections import OrderedDict

class TestRatioBetweenColumns(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [10,10,9],
                "v2": [1,2,3],
                "v3": [3,2,1]
            })
        )

    def test_defaults(self):
        op = dp.RatioBetweenColumns(numer='v1', denom='v2')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [10,10,9],
                "v2": [1,2,3],
                "v3": [3,2,1],
                "v1Tov2Ratio": [10., 5., 3.]
            })
        )
        assert_frame_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.RatioBetweenColumns(numer='v1', denom='v2', feat_name='my_feat')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [10,10,9],
                "v2": [1,2,3],
                "v3": [3,2,1],
                "my_feat": [10., 5., 3.]
            })
        )
        assert_frame_equal(res, expected_res)

    def test_Errors(self):
        self.assertRaises(ValueError, dp.RatioBetweenColumns)
        self.assertRaises(ValueError,
                          dp.RatioBetweenColumns(
                              numer='foo', denom='bar').transform,([1,2,3]))


class TestRatioColumnToConst(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [3,6,9],
                "v2": [1,2,3],
                "v3": [3,2,1]
            })
        )

    def test_defaults(self):
        op = dp.RatioColumnToConst(col='v1', const=3)
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [3,6,9],
                "v2": [1,2,3],
                "v3": [3,2,1],
                "v1To3Ratio": [1., 2., 3.]
            })
        )
        assert_frame_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.RatioColumnToConst(col='v1', const=3, feat_name='my_feat')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [3,6,9],
                "v2": [1,2,3],
                "v3": [3,2,1],
                "my_feat": [1., 2., 3.]
            })
        )
        assert_frame_equal(res, expected_res)

    def test_Errors(self):
        self.assertRaises(ValueError, dp.RatioColumnToConst)
        self.assertRaises(ValueError, dp.RatioColumnToConst, const=3)
        self.assertRaises(ValueError,
                          dp.RatioColumnToConst(col='a', const=4).transform,[1,2,3])


class TestRatioColumnToValue(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 1, 2, 2, 4],
                "v2": [1, 2, 2, 4, 10]
            })
        )
        self.df2 = pd.DataFrame(OrderedDict(
            {
                "v1": [10, 20, 30, 40, 50],
                "v2": [10, 20, 30, 40, 50]
            }
        ))

    def test_defaults_mean(self):
        op = dp.RatioColumnToValue(col='v1', func='mean')
        res = op.fit_transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 1, 2, 2, 4],
                "v2": [1, 2, 2, 4, 10],
                "v1_RatioTo_mean": [0.5, 0.5, 1., 1., 2.]
            })
        )
        assert_frame_equal(res, expected_res)

        res2 = op.transform(self.df2)
        expected_res2 = pd.DataFrame(OrderedDict(
            {
                "v1": [10, 20, 30, 40, 50],
                "v2": [10, 20, 30, 40, 50],
                "v1_RatioTo_mean": [5., 10., 15., 20., 25.]
            }
        ))
        assert_frame_equal(res2, expected_res2)

    def test_defaults_median(self):
        op = dp.RatioColumnToValue(col='v2', func='median')
        res = op.fit_transform(self.df)

        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 1, 2, 2, 4],
                "v2": [1, 2, 2, 4, 10],
                "v2_RatioTo_median": [0.5, 1., 1., 2., 5.]
            })
        )
        assert_frame_equal(res, expected_res)

        res2 = op.transform(self.df2)
        expected_res2 = pd.DataFrame(OrderedDict(
            {
                "v1": [10, 20, 30, 40, 50],
                "v2": [10, 20, 30, 40, 50],
                "v2_RatioTo_median": [5., 10., 15., 20., 25.]
            }
        ))
        assert_frame_equal(res2, expected_res2)


class TestDaysFromLaterToEarly(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [1,2,3],
                "start": [
                    pd.datetime(2017,1,1),
                    pd.datetime(2016,12,12),
                    pd.datetime(2017,1,10)
                ],
                "end": [
                    pd.datetime(2017,1,10),
                    pd.datetime(2016,12,15),
                    pd.datetime(2017,1,10)
                ]
            })
        )

    def test_defaults(self):
        op = dp.DaysFromLaterToEarly(start='start', end='end')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1,2,3],
                "start": [
                    pd.datetime(2017,1,1),
                    pd.datetime(2016,12,12),
                    pd.datetime(2017,1,10)
                ],
                "end": [
                    pd.datetime(2017,1,10),
                    pd.datetime(2016,12,15),
                    pd.datetime(2017,1,10)
                ],
                "DaysFrom_start_To_end": pd.Series([9, 3, 0])
            })
        )
        assert_frame_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.DaysFromLaterToEarly(start='start', end='end', feat_name='foo')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1,2,3],
                "start": [
                    pd.datetime(2017,1,1),
                    pd.datetime(2016,12,12),
                    pd.datetime(2017,1,10)
                ],
                "end": [
                    pd.datetime(2017,1,10),
                    pd.datetime(2016,12,15),
                    pd.datetime(2017,1,10)
                ],
                "foo": pd.Series([9, 3, 0])
            })
        )
        assert_frame_equal(res, expected_res)

    def test_Errors(self):
        self.assertRaises(ValueError, dp.DaysFromLaterToEarly),
        self.assertRaises(ValueError,
                          dp.DaysFromLaterToEarly(start='foo', end='bar').transform,
                          [1,2,4])


class TestDayOfTheWeekForColumn(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27),
                    pd.datetime(2017, 6, 26),
                    pd.datetime(2017, 6, 25)
                ]
            }
        )
        )

    def test_defaults(self):
        op = dp.DayOfTheWeekForColumn(col='date')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27),
                    pd.datetime(2017, 6, 26),
                    pd.datetime(2017, 6, 25)
                ],
                "date_DayOfTheWeek": pd.Series([1,0,6]).astype('category')
            }
        ))
        assert_frame_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.DayOfTheWeekForColumn(col='date', feat_name='foo')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27),
                    pd.datetime(2017, 6, 26),
                    pd.datetime(2017, 6, 25)
                ],
                "foo": pd.Series([1,0,6]).astype('category')
            }
        ))
        assert_frame_equal(res, expected_res)

    def test_Errors(self):
        self.assertRaises(ValueError, dp.DayOfTheWeekForColumn)
        self.assertRaises(ValueError,
                          dp.DayOfTheWeekForColumn(col='foo').transform,
                          [1,2,3])


class TestHourOfTheDayForColumn(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27, 10, 12, 52),
                    pd.datetime(2017, 6, 26, 23, 0, 12),
                    pd.datetime(2017, 6, 25, 8, 0, 0)
                ]
            }
        )
        )

    def test_defaults(self):
        op = dp.HourOfTheDayForColumn(col='date')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27, 10, 12, 52),
                    pd.datetime(2017, 6, 26, 23, 0, 12),
                    pd.datetime(2017, 6, 25, 8, 0, 0)
                ],
                "date_HourOfTheDay": pd.Series([10,23,8]).astype('category')
            }
        ))
        assert_frame_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.HourOfTheDayForColumn(col='date', feat_name='foo')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [1, 2, 3],
                "date": [
                    pd.datetime(2017, 6, 27, 10, 12, 52),
                    pd.datetime(2017, 6, 26, 23, 0, 12),
                    pd.datetime(2017, 6, 25, 8, 0, 0)
                ],
                "foo": pd.Series([10, 23, 8]).astype('category')
            }
        ))
        assert_frame_equal(res, expected_res)

    def test_Errors(self):
        self.assertRaises(ValueError, dp.HourOfTheDayForColumn)
        self.assertRaises(ValueError,
                          dp.HourOfTheDayForColumn(col='foo').transform,
                          [1,2,3])


class TestSelectColumns(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            np.random.random(size=(10,5)),
            columns=['foo', 'bar', 'goo', 'hello', 'world']
        )

    def test_defaults(self):
        op = dp.SelectColumns(cols=['foo', 'bar'])
        res = op.transform(self.df)

        np.random.seed(42)
        expected_res = pd.DataFrame(
            np.random.random(size=(10,5)),
            columns=['foo', 'bar', 'goo', 'hello', 'world']
        )[['foo', 'bar']]
        assert_frame_equal(res, expected_res)
