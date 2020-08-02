# -*- coding: utf-8 -*-
from .version import __version__

from .config import CONFIG, get_config, set_config
from .data import DataPairs
from .datasets import Dataset
from .datasets.standard import get_standard_dataset
from .reference_reconstructors import get_reference_reconstructor
from .reconstructors import (
    Reconstructor, IterativeReconstructor,
    StandardIterativeReconstructor, LearnedReconstructor)
from .measure import Measure
from .evaluation import TaskTable


__all__ = ['CONFIG', 'get_config', 'set_config',
           'DataPairs',
           'Dataset', 'get_standard_dataset',
           'get_reference_reconstructor',
           'Reconstructor', 'IterativeReconstructor',
           'StandardIterativeReconstructor', 'LearnedReconstructor',
           'Measure',
           'TaskTable']

try:
    from .reconstructors import StandardLearnedReconstructor
except ImportError:
    pass
else:
    __all__.append('StandardLearnedReconstructor')
