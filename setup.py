# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='dival',
      version='0.1.2',
      description='Deep Inversion Validation Library',
      url='https://gitlab.informatik.uni-bremen.de/dival/dival',
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
