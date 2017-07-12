import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from pubdsutils import features_engineering as fe
from collections import OrderedDict
from sklearn.pipeline import make_pipeline


class TestSimplePipeline(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(OrderedDict(
            {
                "v1": [2, 4, 6],
                "v2": [10, 20, 30],
                "d1": [
                    pd.datetime(2017, 6, 27),
                    pd.datetime(2017, 6, 24),
                    pd.datetime(2017, 5, 1)
                ],
                "d2": [
                    pd.datetime(2017, 6, 20),
                    pd.datetime(2017, 6, 20),
                    pd.datetime(2017, 4, 30)
                ]
            })
        )

    def test_simple_pipeline(self):
        df_copy = self.df.copy()
        pipeline = make_pipeline(
            fe.RatioColumnToConst(col='v1', const=2, feat_name='v1_to_2'),
            fe.RatioBetweenColumns(
                numer='v2', denom='v1', feat_name='v2_to_v1')
        )
        res = pipeline.transform(self.df)
        expected_res = pd.DataFrame(OrderedDict(
            {
                "v1": [2, 4, 6],
                "v2": [10, 20, 30],
                "d1": [
                    pd.datetime(2017, 6, 27),
                    pd.datetime(2017, 6, 24),
                    pd.datetime(2017, 5, 1)
                ],
                "d2": [
                    pd.datetime(2017, 6, 20),
                    pd.datetime(2017, 6, 20),
                    pd.datetime(2017, 4, 30)
                ],
                'v1_to_2': [1., 2., 3.],
                'v2_to_v1': [5., 5., 5.]
            }
        ))
        assert_frame_equal(res, expected_res)
        assert_frame_equal(self.df, df_copy)
