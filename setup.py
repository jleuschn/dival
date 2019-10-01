# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from dival import __version__

setup(name='dival',
      version=__version__,
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
          'matplotlib'
      ],
      zip_safe=False)
