import pandas as pd
from hashlib import sha256
import pymssql
import configparser


def from_sql_sever(config_file, query=None):
    """
    Fetch data from MS SQL

    Query the MS SQL data warehouse using ``query`` and the settings
    in the configuration file.
    Following is an example of the config file as ``.ini``.

    .. code-block:: ini

        [Base]
        server = url.of.server
        domain = DOMAIN
        username = user.name
        password = yourpassword

    *Note* that this ``.ini`` contains your password! Be careful.

    Parameters
    ----------
    config_file : str
        location of the configuration file
    query: str
        valid SQL query as a string

    Returns
    -------
    pandas.DataFrame
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    conn = pymssql.connect(
        config['Base']['server'],
        config['Base']['domain'] + '\\' + config['Base']['username'],
        config['Base']['password'],
        port=1433
    )
    if query is None:
        raise ValueError("query must be provided")

    df = pd.read_sql(query, conn)
    return df


def persist_df(df, path=None, sql=None, prefix='raw_df'):
    """
    Pickles a DataFrame and assigns hash to the filename

    Optionally, can also include the related SQL query

    Parameters
    ----------
    path : str or None (default)
        If not None should be a path including a trailing ``/``.
        If ``None``, use the current directory.
    sql : str (optional)
        If provided, string of the SQL which generated the DataFrame will
        be stored alongside the pickle.
    prefix : str (default ``raw_df``)
        Prefix of pickels
    """
    df_hash = sha256(df.to_json().encode()).hexdigest()
    base_filename = '{}_{}_{}'.format(
        prefix,
        pd.datetime.now().isoformat().replace(":", "-").replace(".", "-"),
        df_hash)
    if path is not None:
        base_filename = path + base_filename
    df.to_pickle(base_filename + '.pickle')
    with open(base_filename + ".sql", "w") as sql_file:
        print(sql, file=sql_file)
