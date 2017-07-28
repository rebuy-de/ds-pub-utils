"""
Preprocessing utilities for pandas.DataFrame_ which plays nicely with
scikit-learn

.. _pandas.DataFrame : https://is.gd/GdHbXc
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted
import pandas as pd

import pubdsutils as pdu


class RemoveConstantColumns(TransformerMixin):
    """
    Identify constant columns and enable their removal

    Attributes
    ----------
    const_cols : list
        The list of column names which are constant.
        List may be of length 1.
    """

    def transform(self, df, **transform_params):
        """
        Returns a copy of ``df`` where the constant columns (as identified)
        during the ``fit`` are removed

        Parameters
        ----------
        df : DataFrame
            Should have the same columns as those of the DataFrame used at
            fitting.
        """
        check_is_fitted(self, 'const_cols')
        return df.drop(self.const_cols, axis=1)

    def fit(self, df, y=None, **fit_params):
        """
        Identify the constant columns

        Parameters
        ----------
        df : DataFrame
            Data from which constant features are identified
        """
        self.const_cols = df.loc[:, df.apply(pd.Series.nunique) == 1].columns
        return self


class ColumnsOneHotEncoder(BaseEstimator, TransformerMixin):
    """Batch One-Hot-Encode a set of columns, all with the same number of values

    This class is designed to be used when the DataFrame contains one or more
    columns containing categorical data of the same type. For example:

    - Order's day of the week
    - Shipping's day of the week.

    In this example, both columns have 7 possible values, and can/should be
    encoded using OneHotEncoder. This class assists in doing so when having the
    data as a `pandas.DataFrame`.

    Attributes
    ----------
    cols : list
        List of categorical columns to be One-Hot-Encoded.
    n_values : int
        Number of values (see sklearn.preprocessing.OneHotEncoder_ for
        more details)

        .. _sklearn.preprocessing.OneHotEncoder : https://is.gd/Qi1YFg
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
        """
        Returns a copy of ``df`` where ``cols`` are replaced with their
        One-Hot-Encoding

        Parameters
        ----------
        df : DataFrame
            DataFrame to transform
        """
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
        """
        Fitting the instance on ``df``

        Parameters
        ----------
        df : DataFrame
            The base DataFrame from which column names are learned.
            These names will be used when transforming data
        """
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        self.ohe.fit(df[self.cols])
        self.ohe_cols_names_ = []

        for col in self.cols:
            for i in range(self.n_values):
                self.ohe_cols_names_.append(col + '_' + str(i))

        return self


class StandardizeFloatCols(BaseEstimator, TransformerMixin):
    """Standard-scale the columns in the data frame.


    Apply sklearn.preprocessing.StandardScaler_ to `cols`

    .. _sklearn.preprocessing.StandardScaler : https://is.gd/cdMuLr

    Attributes
    ----------
    cols : list
        List of columns in the data to be scaled
    """

    def __init__(self, cols=None):
        pdu._is_cols_input_valid(cols)
        self.cols = cols
        self.standard_scaler = StandardScaler()
        self._is_fitted = False

    def transform(self, df, **transform_params):
        """
        Scaling ``cols`` of ``df`` using the fitting

        Parameters
        ----------
        df : DataFrame
            DataFrame to be preprocessed
        """
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
        """
        Fitting the preprocessing

        Parameters
        ----------
        df : DataFrame
            Data to use for fitting.
            In many cases, should be ``X_train``.
        """
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        self.standard_scaler.fit(df[self.cols])
        self._is_fitted = True
        return self


class LabelEncodingColoumns(BaseEstimator, TransformerMixin):
    """Label encoding selected columns

    Apply sklearn.preprocessing.LabelEncoder_ to `cols`

    .. _sklearn.preprocessing.LabelEncoder : https://is.gd/Vx2njl

    Attributes
    ----------
    cols : list
        List of columns in the data to be scaled
    """

    def __init__(self, cols=None):
        pdu._is_cols_input_valid(cols)
        self.cols = cols
        self.les = {col: LabelEncoder() for col in cols}
        self._is_fitted = False

    def transform(self, df, **transform_params):
        """
        Label encoding ``cols`` of ``df`` using the fitting

        Parameters
        ----------
        df : DataFrame
            DataFrame to be preprocessed
        """
        if not self._is_fitted:
            raise NotFittedError("Fitting was not preformed")
        pdu._is_cols_subset_of_df_cols(self.cols, df)

        df = df.copy()

        label_enc_dict = {}
        for col in self.cols:
            label_enc_dict[col] = self.les[col].transform(df[col])

        labelenc_cols = pd.DataFrame(label_enc_dict,
            # The index of the resulting DataFrame should be assigned and
            # equal to the one of the original DataFrame. Otherwise, upon
            # concatenation NaNs will be introduced.
            index=df.index
        )

        for col in self.cols:
            df[col] = labelenc_cols[col]
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Fitting the preprocessing

        Parameters
        ----------
        df : DataFrame
            Data to use for fitting.
            In many cases, should be ``X_train``.
        """
        pdu._is_cols_subset_of_df_cols(self.cols, df)
        for col in self.cols:
            self.les[col].fit(df[col])
        self._is_fitted = True
        return self
