import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from pubdsutils import data_preparation as dp
from collections import OrderedDict

class TestRatioBetweenColumnsDF(unittest.TestCase):

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


class TestRatioBetweenColumnsSeries(unittest.TestCase):

    def setUp(self):
        self.ser = pd.Series(OrderedDict(
            {
                "v1": 10,
                "v2": 2,
                "v3": 3,
            })
        )

    def test_defaults(self):
        op = dp.RatioBetweenColumns(numer='v1', denom='v2')
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 10,
                "v2": 2,
                "v3": 3,
                "v1Tov2Ratio": 5.
            })
        )
        assert_series_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.RatioBetweenColumns(numer='v1', denom='v2', feat_name='my_feat')
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 10,
                "v2": 2,
                "v3": 3,
                "my_feat": 5.
            })
        )
        assert_series_equal(res, expected_res)


class TestRatioColumnToConstDF(unittest.TestCase):

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
        self.assertRaises(ValueError,
                          dp.RatioColumnToConst(col='a', const=4).transform,[1,2,3])


class TestRatioColumnToConstSeries(unittest.TestCase):

    def setUp(self):
        self.ser = pd.Series(OrderedDict(
            {
                "v1": 6,
                "v2": 1,
                "v3": 3
            })
        )

    def test_defaults(self):
        op = dp.RatioColumnToConst(col='v1', const=3)
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 6,
                "v2": 1,
                "v3": 3,
                "v1To3Ratio": 2.
            })
        )
        assert_series_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.RatioColumnToConst(col='v1', const=3, feat_name='my_feat')
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 6,
                "v2": 1,
                "v3": 3,
                "my_feat": 2.
            })
        )
        assert_series_equal(res, expected_res)


class TestDaysFromLaterToEarlyDF(unittest.TestCase):

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


class TestDaysFromLaterToEarlySeries(unittest.TestCase):

    def setUp(self):
        self.ser = pd.Series(OrderedDict(
            {
                "v1": 2,
                "start": pd.datetime(2017,1,1),
                "end": pd.datetime(2017,1,10)
            })
        )
        self

    def test_defaults(self):
        op = dp.DaysFromLaterToEarly(start='start', end='end')
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 2,
                "start": pd.datetime(2017,1,1),
                "end": pd.datetime(2017,1,10),
                "DaysFrom_start_To_end": 9
            })
        )
        assert_series_equal(res, expected_res)

    def test_set_feat_name(self):
        op = dp.DaysFromLaterToEarly(start='start', end='end', feat_name='foo')
        res = op.transform(self.ser)
        expected_res = pd.Series(OrderedDict(
            {
                "v1": 2,
                "start": pd.datetime(2017,1,1),
                "end": pd.datetime(2017,1,10),
                "foo": 9
            })
        )
        assert_series_equal(res, expected_res)


class TestDayOfTheWeekForColumnDF(unittest.TestCase):

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


class TestDayOfTheWeekForColumnSeries(unittest.TestCase):

    def setUp(self):
        self.df = pd.Series(OrderedDict(
            {
                "v1": 1,
                "date": pd.datetime(2017, 6, 27)
            }
        ))

    def test_defaults(self):
        op = dp.DayOfTheWeekForColumn(col='date')
        res = op.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": 1,
                "date": pd.datetime(2017, 6, 27),
                "date_DayOfTheWeek": pd.Series([1]).astype('category')
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
