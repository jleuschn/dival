# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
with open(os.path.join(os.path.split(__file__)[0], 'VERSION')) as version_f:
    version = version_f.read().strip()

setup(name='dival',
      version=version,
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
