from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import pubdsutils as pdu


class RatioBetweenColumns(BaseEstimator, TransformerMixin):
    """
    Ratio between two columns

    Adds a column (or attribute) holding the ratio between
    two columns of a DataFrame (or between two attributes of a Series).

    Returns a *copy* of the input.

    Attributes
    ----------
    numer : str
        Column name of the numerator
    denom : str
        Column name of the denominator
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``numerTOdenomRatio`` will be used
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
        """
        Returns a copy of ``df`` with a new column (named ``feat_name``) and
        holding the ratio between ``df.numer`` and ``df.denom``.
        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.numer].div(df[self.denom])
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        return self


class RatioColumnToConst(BaseEstimator, TransformerMixin):
    """
    Ratio between a column and a constant

    Adds a column (or attribute) holding the ration between a column of
    a DataFrame (or an attribute of a Series) and a constant value

    Returns a *copy* of the input.

    Attributes
    ----------
    col : str
        Column name of the numerator
    const : numerical value
        Value of the denominator
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``colTOconstRatio`` will be used
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
        """
        Returns a copy of ``df`` with a new column (named ``feat_name``) and
        holding the ratio between ``df.col`` and ``const``.
        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = df[self.col].div(self.const)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        return self


class RatioColumnToValue(BaseEstimator, TransformerMixin):
    """
    Compute the ratio between the values in a column to either the `mean`
    or `median`

    An instance has to fitted; at this step the ``func`` is applied to the
    column ``col`` and the result is stored in ``cosnt_``.
    Then, ``const_`` is used to computed the ratio.

    Assume you have ``X_train`` and ``y_train``.
    Furthermore, ``X_train`` has a feature ``val``.
    Say that you want to obtain a new feature which is the ratio between
    ``X_train.val`` its mean.
    When preparing ``X_test``, the new feature has to be computed and the mean
    fitted from ``X_train`` has to be used.
    This class, bridges this gap.

    .. code-block:: python

        rtc = RatioColumnToValue(col='val', func='mean')
        rtc.fit(X_train)
        rtc.transform(X_test)


    Attributes
    ----------

    col : str
        Column name of the feature
    func : str
        Function to use. Support either `mean` or `median`
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``col_RatioTo_func`` will be used
    """

    def __init__(self, col=None, func=None, feat_name=None):
        if col is None or func is None:
            raise ValueError("Both col and func have to be provided")
        self.col = col
        self.func = func
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_RatioTo_{}".format(self.col, self.func)

    def transform(self, df, **transform_params):
        """
        Compute the ratio between ``df[col]`` and the mean/median which
        was fitted.
        """
        check_is_fitted(self, 'const_')
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pdu._is_cols_subset_of_df_cols([self.col], df)
            df[self.feat_name] = df[self.col].div(self.const_)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Fits the instance to the mean/median of ``df[col]``
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Non supported input")
        pdu._is_cols_subset_of_df_cols([self.col], df)
        if self.func == 'mean':
            self.const_ = df[self.col].mean()
        elif self.func == 'median':
            self.const_ = df[self.col].median()
        else:
            raise ValueError(
                "Unsupported function ({}). Can be either mean or "
                "median".format(self.func)
            )
        return self


class DaysFromLaterToEarly(BaseEstimator, TransformerMixin):
    """
    Compute number of days between two time features

    Attributes
    ----------
    start : str
        Column name of the earlier (start) date
    end : str
        Column name of the later (end) date
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``DaysFrom_start_To_end`` will be used
    """

    def __init__(self, start=None, end=None, feat_name=None):
        if start is None or end is None:
            raise ValueError("Both start and end have to be specified")
        self.start = start
        self.end = end
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "DaysFrom_{}_To_{}".format(self.start, self.end)

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` with a new column (named ``feat_name``) and
        holding the number of days passed between ``df.start`` and ``df.end``.
        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            df[self.feat_name] = (df[self.end] - df[self.start]).apply(
                lambda x: x.days)
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        return self


class DayOfTheWeekForColumn(BaseEstimator, TransformerMixin):
    """
    Compute day-of-the-week of a coulmn

    Attributes
    ----------
    col : str
        Column name of the base feature (should be date)
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``col_DayOfTheWeek`` will be used
    """

    def __init__(self, col=None, feat_name=None):
        if col is None:
            raise ValueError("col name must be provided")
        self.col = col
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_DayOfTheWeek".format(col)

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` with a new column (named ``feat_name``) and
        holding the day of the week as derived from ``df.col``.
        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pdu._is_cols_subset_of_df_cols([self.col], df)
            df[self.feat_name] = df[self.col].apply(
                lambda x: x.dayofweek).astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        pdu._is_cols_subset_of_df_cols([self.col], df)
        return self


class HourOfTheDayForColumn(BaseEstimator, TransformerMixin):
    """
    Compute hour-of-the-day of a coulmn

    Attributes
    ----------
    col : str
        Column name of the base feature (should be date)
    feat_name : str (default None)
        Column name of the new featrue. If ``None`` a default name
        ``col_HourOfTheDay`` will be used
    """

    def __init__(self, col=None, feat_name=None):
        if col is None:
            raise ValueError("col must be provided")
        self.col = col
        self.feat_name = feat_name
        if self.feat_name is None:
            self.feat_name = "{}_HourOfTheDay".format(col)

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` with a new column (named ``feat_name``) and
        holding the hour of the day as derived from ``df.col``.
        """
        df = df.copy()
        if isinstance(df, pd.DataFrame):
            pdu._is_cols_subset_of_df_cols([self.col, ], df)
            df[self.feat_name] = df[self.col].apply(
                lambda x: x.hour).astype('category')
        else:
            raise ValueError("Non supported input")
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        pdu._is_cols_subset_of_df_cols([self.col, ], df)
        return self


class SelectColumns(BaseEstimator, TransformerMixin):
    """
    Selects the columns/features

    Attributes
    ----------
    cols : list
        List of column names to be selected
    """

    def __init__(self, cols=None):
        pdu._is_cols_input_valid(cols)
        self.cols = cols

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` holding the columns defined in ``cols``
        """
        # TODO add a test that self.cols is a subset of the columns if df
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        return df[self.cols].copy()

    def fit(self, df, y=None, **fit_params):
        """
        Doesn't do anything.
        Provided for the sake of consistency with scikit-learn.
        """
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        return self
