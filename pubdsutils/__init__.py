def _is_cols_subset_of_df_cols(cols, df):
    if set(cols).issubset(set(df.columns)):
        return True
    else:
        raise ValueError(
            "Class instantiated with columns that don't appear in the data frame"
        )

def _is_cols_input_valid(cols):
    if cols is None or not isinstance(cols, list) or len(cols) == 0 or not all(isinstance(col, str) for col in cols):
        raise ValueError(
            "Cols should be a list of strings. Each string should correspond to a column name")
    else:
        return True
