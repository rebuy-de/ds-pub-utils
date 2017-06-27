from sklearn.base import BaseEstimator, TransformerMixin
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
        elif isinstance(df, pd.Series):
            df[self.feat_name] = df[self.numer] / df[self.denom]
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
        elif isinstance(df, pd.Series):
            df[self.feat_name] = df[self.col] / self.const
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
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
        print(type(df))
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = (df[self.end] - df[self.start]).apply(lambda x: x.days)
        elif isinstance(df, pd.Series):
            df[self.feat_name] = (df[self.end] - df[self.start]).days
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
        elif isinstance(df, pd.Series):
            df[self.feat_name] = df[self.col].dayofweek.astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class HourOfTheDayForColumn(BaseEstimator, TransformerMixin):

    def __init__(self, col=None, feat_name=None):
        self.col = col
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}-HourOfTheDay".format(col)

    def transform(self, df, **transform_params):
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].apply(
                lambda x: x.hour).astype('category')
        elif isinstance(df, pd.Series):
            df[self.feat_name] = df[self.col].hour.astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        return self


class SelectColumns(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df, **transform_params):
        return df[self.cols].copy()

    def fit(self, df, y=None, **fit_params):
        return self
