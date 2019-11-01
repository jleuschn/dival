# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

VERSION = '0.3.4'
with open(os.path.join(os.path.split(__file__)[0], 'dival',
                       'VERSION'), 'w') as version_f:
    version_f.write(VERSION)

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
