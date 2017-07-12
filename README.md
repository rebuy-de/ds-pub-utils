[![logo](./docs/img/reBuy-logo.png)](http://www.rebuy.com)

------

# Public Data Science Utilities @ [reBuy](http://www.rebuy.com)

## Warning / License

This package is, by and large, under active development and *nothing* should be taken here for granted.
It is intended to be used as part of other, internal, workflows.
Therefore, it is *very* likely that changes will occur.
It is available under the [MIT license](./license.md).

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

0. (Optional but recommended) Start a new virtual environment. For example using `conda create --name test-this python=3`
1. Clone the repository
2. Run `python setup.py install`. To be on a safer side, install using `python setup.py install --record files.txt`. The produced `files.txt` can be later used for [uninstalling the package](https://stackoverflow.com/a/1550235/671013).
