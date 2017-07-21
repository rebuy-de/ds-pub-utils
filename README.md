[![logo](./docs/img/reBuy-logo.png)](http://www.rebuy.com)

------

# Public Data Science Utilities @ [reBuy](http://www.rebuy.com)

Write to us: datascience@rebuy.com

## Warning / License

This package is, by and large, under active development and *nothing* should be taken here for granted.
It is intended to be used as part of other, internal, workflows.
Therefore, it is *very* likely that changes will occur.
It is available under the [MIT license](./license.md).

Lastly, this is a public repository; _**DO NOT INCLUDE ANY BUSINESS LOGIC NOR DATA NOR ANYTHING CONFIDENTIAL!**_

## Provided Modules

### `features_engineering`

Along the design lines of [`Scikit learn`](http://scikit-learn.org/), the classes in this module provide `fit`, `transform` and `fit_transform` functionalities.
However, when transforming data using these classes, _new_ features are _added_, and nothing is removed.

### `preprocessing`

Inspired by [sklearn-pandas](https://github.com/pandas-dev/sklearn-pandas), this module provide preprocessing functionalities for columns of a DataFrame.
In contrast to the features engineering module, this one doesn't append columns to the data, but rather replaces.

### `data_fetch`

Utilities for data fetching

## Installation

0. (Optional but recommended) Start a new virtual environment.
    1. Either using `conda create --name test-this python=3`. The package needs Python 3.x.
    2. Or, use the provided `environment.yml`.
1. Clone the repository
2. Run `pip install -e .` from the directory of the package
3. (Optional) you can run `pytest` from the root of the package and see if all tests passes

Remark: The function `data_fetch.from_sql_sever` uses [`pymssql`](http://pymssql.org/en/stable/intro.html#install) which in turn depends on  [`freetds`](http://pymssql.org/en/stable/freetds.html).
If you want to use this function, make sure you install `pymssql`.
[This SO thread](https://stackoverflow.com/q/17368964/671013) might be helpful as well

## Uninstallation

At `{virtualenv}/lib/python2.7/site-packages/` (if not using `virtualenv` then `{system_dir}/lib/python2.7/dist-packages/`) remove the egg file (e.g. `pubdsutils-0.6.34-py2.7.egg`) if there is any.
From file `easy-install.pth`, remove the corresponding line (it should be a path to the source directory or of an egg file).
Source is [SO answer](https://stackoverflow.com/a/18818891/671013).

# Maintaining issues:

* Use `flake8 --exclude=build` to check that the code is well styled
* Use `pytest --cov-report term-missing --cov=pubdsutils tests/` to check the tests coverage
* Execute `sphinx-apidoc -f -o . ../pubdsutils/` from `./docs` when adding/removing module/packages
* `make html` from `./docs` will generate the documentation
