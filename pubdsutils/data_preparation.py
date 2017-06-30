from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

class RatioBetweenColumns(BaseEstimator, TransformerMixin):
    """Adds a column (or attribute) holding the ratio between
    two columns of a DataFrame (or between two attributes of a Series)

    Returns a *copy* of the given input.
    """

    def __init__(self, numer=None, denom=None, feat_name=None):
        if numer is None or denom is None:
            raise ValueError("Both numer and denom have to be specified")
        self.numer = numer
        self.denom = denom
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}To{}Ratio".format(numer, denom)

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.numer].div(df[self.denom])
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class RatioColumnToConst(BaseEstimator, TransformerMixin):
    """Adds a column (or attribute) holding the ration between a column of
    a DataFrame (or an attribute of a Series) and a constant value

    Returns a *copy* of the given input.
    """

    def __init__(self, col=None, const=None, feat_name=None):
        if col is None or const is None:
            raise ValueError("Both col and const have to be specified")
        self.col = col
        self.const = const
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}To{}Ratio".format(self.col, str(self.const))

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].div(self.const)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class RatioColumnToValue(BaseEstimator, TransformerMixin):
    """Compute the ratio between the values in a column to either its `mean`
    or `median`.
    """

    def __init__(self, col=None, func=None, feat_name=None):
        if col is None and func is None:
            raise ValueError("Both col and func have to be provided")
        self.col = col
        self.func = func
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_RatioTo_{}".format(self.col, self.func)

    def transform(self, df, **transform_params):
        check_is_fitted(self, 'const_')
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].div(self.const_)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        if self.func == 'mean':
            self.const_ = df[self.col].mean()
        elif self.func == 'median':
            self.const_ = df[self.col].median()
        else:
            raise ValueError(
                "Unsupported function ({}). Can be either mean or median"
                ).format(self.func)
        return self


class DaysFromLaterToEarly(BaseEstimator, TransformerMixin):

    def __init__(self, start=None, end=None, feat_name=None):
        if start is None or end is None:
            raise ValueError("Both start and end have to be specified")
        self.start = start
        self.end = end
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "DaysFrom_{}_To_{}".format(self.start, self.end)

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = (df[self.end] - df[self.start]).apply(lambda x: x.days)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class DayOfTheWeekForColumn(BaseEstimator, TransformerMixin):

    def __init__(self, col=None, feat_name=None):
        if col is None:
            raise ValueError("col name must be provided")
        self.col = col
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_DayOfTheWeek".format(col)

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].apply(
                lambda x: x.dayofweek).astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class HourOfTheDayForColumn(BaseEstimator, TransformerMixin):

    def __init__(self, col=None, feat_name=None):
        if col is None:
            raise ValueError("col must be provided")
        self.col = col
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_HourOfTheDay".format(col)

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].apply(
                lambda x: x.hour).astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class SelectColumns(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        if cols is None:
            raise ValueError("List of columns cols must be provided. Currently None")
        self.cols = cols

    def transform(self, df, **transform_params):
        return df[self.cols].copy()

    def fit(self, df, y=None, **fit_params):
        return self
