# -*- coding: utf-8 -*-
import os
with open(os.path.join(os.path.split(__file__)[0], 'VERSION')) as version_f:
    __version__ = version_f.read().strip()

from .config import CONFIG, get_config, set_config
from .data import DataPairs
from .datasets import Dataset
from .datasets.standard import get_standard_dataset
from .reconstructors import (Reconstructor, IterativeReconstructor,
                             LearnedReconstructor)
from .measure import Measure
from .evaluation import TaskTable


__all__ = ['CONFIG', 'get_config', 'set_config',
           'DataPairs',
           'Dataset', 'get_standard_dataset',
           'Reconstructor', 'IterativeReconstructor', 'LearnedReconstructor',
           'Measure',
           'TaskTable']
