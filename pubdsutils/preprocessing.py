from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

import pubdsutils as pdu


class RemoveConstantColumns(TransformerMixin):

    def transform(self, df, **transform_params):
        check_is_fitted(self, 'const_cols')
        return df.drop(self.const_cols, axis=1)

    def fit(self, df, y=None, **fit_params):
        self.const_cols = df.loc[:, df.apply(pd.Series.nunique) == 1].columns
        return self


class ColumnsOneHotEncoder(BaseEstimator, TransformerMixin):
    """Batch One-Hot-Encode a collection of columns, all sharing the same number of values

    This class is designed to be used when the DataFrame contains one or more columns
    containing categorical data of the same type. For example:

    - Order's day of the week
    - Shipping's day of the week.

    In this example, both columns have 7 possible values, and can/should be
    encoded using OneHotEncoder. This class assists in doing so when having the
    data as a `pandas.DataFrame`.
    """

    def __init__(self, cols=None, n_values=None):
        if cols is None or n_values is None:
            raise ValueError("Both cols and n_values have to be specified")
        if not isinstance(n_values, int):
            raise ValueError("n_values should be an integer")
        pdu._is_cols_input_valid(cols)
        self.cols = cols
        self.n_values = n_values
        self.ohe = OneHotEncoder(n_values=self.n_values)

    def transform(self, df, y=None, **trans_param):
        check_is_fitted(self, 'ohe_cols_names_')
        df = df.copy()
        ohe_cols_arr = self.ohe.transform(df[self.cols]).toarray()
        ohe_cols_df = pd.DataFrame(
            ohe_cols_arr,
            columns=self.ohe_cols_names_,
            index=df.index
        )
        return pd.concat([df.drop(self.cols, axis=1), ohe_cols_df], axis=1)

    def fit(self, df, y=None, **fit_params):
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        self.ohe.fit(df[self.cols])
        self.ohe_cols_names_ = []

        for col in self.cols:
            for i in range(self.n_values):
                self.ohe_cols_names_.append(col + '_' + str(i))

        return self


class StandartizeFloatCols(BaseEstimator, TransformerMixin):
    """Standard-scale the columns in the data frame.

    `cols` should be a list of columns in the data.
    """

    def __init__(self, cols=None):
        pdu._is_cols_input_valid(cols)
        self.cols = cols
        self.standard_scaler = StandardScaler()
        self._is_fitted = False

    def transform(self, df, **transform_params):
        if not self._is_fitted:
            raise NotFittedError("Fitting was not preformed")
        pdu._is_cols_subset_of_df_cols(self.cols, df)

        df = df.copy()

        standartize_cols = pd.DataFrame(
            # StandardScaler returns a NumPy.array, and thus indexing
            # breaks. Explicitly fixed next.
            self.standard_scaler.transform(df[self.cols]),
            columns=self.cols,
            # The index of the resulting DataFrame should be assigned and
            # equal to the one of the original DataFrame. Otherwise, upon
            # concatenation NaNs will be introduced.
            index=df.index
        )
        df = df.drop(self.cols, axis=1)
        df = pd.concat([df, standartize_cols], axis=1)
        return df

    def fit(self, df, y=None, **fit_params):
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        self.standard_scaler.fit(df[self.cols])
        self._is_fitted = True
        return self
