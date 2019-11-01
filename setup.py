# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

version = {}
with open("...sample/version.py") as fp:
    exec(fp.read(), version)
VERSION = version['version']

setup(name='dival',
      version=VERSION,
      description='Deep Inversion Validation Library',
      url='https://github.com/jleuschn/dival',
      author='Johannes Leuschner',
      author_email='jleuschn@uni-bremen.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.10',
          'pandas',
          'odl',
          'scikit-image',
          'scikit-learn',
          'hyperopt',
          'pydicom',
          'tqdm',
          'matplotlib',
          'h5py',
          'requests',
          'packaging'
      ],
      include_package_data=True,
      zip_safe=False)
