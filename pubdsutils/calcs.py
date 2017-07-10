import pandas as pd

def value_counts_comb(s, sort=True, ascending=False,
                      bins=None, dropna=True):
    """
    A wrapper of value_counts which returns both the counts and the
    normalized view.

    The resulting object will be in descending order so that the
    first element is the most frequently-occurring element.
    Excludes NA values by default.

    Parameters
    ----------
    normalize : boolean, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.
    sort : boolean, default True
        Sort by values
    ascending : boolean, default False
        Sort in ascending order
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        a convenience for pd.cut, only works with numeric data
    dropna : boolean, default True
        Don't include counts of NaN.

    Returns
    -------
    counts : DataFrame
    """
    from pandas.core.algorithms import value_counts
    from pandas.core.reshape.concat import concat
    res_norm = value_counts(s, sort=sort, ascending=ascending,
                  normalize=True, bins=bins, dropna=dropna)
    res_regu = value_counts(s, sort=sort, ascending=ascending,
                  normalize=False, bins=bins, dropna=dropna)
    result = concat([res_norm, res_regu], axis=1, keys=['Ratio', 'Count'])
    return result
