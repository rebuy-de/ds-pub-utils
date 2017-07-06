def _is_cols_subset_of_df_cols(cols, df):
    """Utility function checking if column(s) in `cols` is/are a subset of the
    the columns of `df`
    """
    if set(cols).issubset(set(df.columns)):
        return True
    else:
        raise ValueError(
            "Class instantiated with columns that don't appear in the data frame"
        )

def _is_cols_input_valid(cols):
    """Utility function checking the validity of the cols parameter.

    The class `features_engineering` and `preprocessing` assume that the
    parameter is:

    - Not `None`
    - Of type list having at least one item
    - All items in the list are strings
    """
    if cols is None or not isinstance(cols, list) or len(cols) == 0 or not all(isinstance(col, str) for col in cols):
        raise ValueError(
            "Cols should be a list of strings. Each string should correspond to a column name")
    else:
        return True
