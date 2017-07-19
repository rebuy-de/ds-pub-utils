from setuptools import setup

setup(name='pubdsutils',
      version='0.1',
      description='Package of DS related public tools from reBuy.com',
      author='reBuy.com',
      author_email='datascience@rebuy.com',
      packages=['pubdsutils'],
      install_requires=[
          'flake8>=3.3.0',
          'numpy>=1.13.0',
          'pandas>=0.20.2',
          'pymssql>=2.1.3',
          'pytest>=3.1.2',
          'pytest-cov>=2.3.1',
          'scikit-learn>=0.18.1',
          'scipy>=0.19.0',
          'pytest'
      ],
      python_requires='>=3',
      zip_safe=False)
