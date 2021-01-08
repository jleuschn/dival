# -*- coding: utf-8 -*-
import os
from warnings import warn
from importlib import import_module
import requests
from dival import get_standard_dataset
from dival.reconstructors import LearnedReconstructor
from dival.config import CONFIG
from dival.util.input import input_yes_no
from dival.util.download import download_file

try:
    DATA_PATH = os.path.normpath(os.path.expanduser(
        CONFIG['reference_params']['data_path']))
except Exception:
    raise RuntimeError(
        'Could not retrieve config value `reference_params/data_path`, '
        'maybe the configuration (e.g. in ~/.dival/config.json) is corrupt.')

# Currently, reference configurations map 1-to-1 to reconstructor types.
# In the future multiple configurations using the same reconstructor type could
# be useful, e.g. 'dip' and 'diptv' both using the type
# `DeepImagePriorCTReconstructor`, but with 'dip' restricted to ``gamma=0.``.
CONFIGURATIONS = {
    'fbp': {
        'type': {'cls': 'FBPReconstructor',
                 'module': 'dival.reconstructors.odl_reconstructors'},
        'datasets': ['ellipses', 'lodopab']},
    'fbpunet': {
        'type': {'cls': 'FBPUNetReconstructor',
                 'module': 'dival.reconstructors.fbpunet_reconstructor'},
        'datasets': ['ellipses', 'lodopab'],
        'learned_params_spec': {'ext': '.pt'}},
    'iradonmap': {
        'type': {'cls': 'IRadonMapReconstructor',
                 'module': 'dival.reconstructors.iradonmap_reconstructor'},
        'datasets': ['ellipses', 'lodopab'],
        'learned_params_spec': {'ext': '.pt'}},
    'learnedgd': {
        'type': {'cls': 'LearnedGDReconstructor',
                 'module': 'dival.reconstructors.learnedgd_reconstructor'},
        'datasets': ['ellipses', 'lodopab'],
        'learned_params_spec': {'ext': '.pt'}},
    'learnedpd': {
        'type': {'cls': 'LearnedPDReconstructor',
                 'module': 'dival.reconstructors.learnedpd_reconstructor'},
        'datasets': ['ellipses', 'lodopab'],
        'learned_params_spec': {'ext': '.pt'}},
    'tvadam': {
        'type': {'cls': 'TVAdamCTReconstructor',
                 'module': 'dival.reconstructors.tvadam_ct_reconstructor'},
        'datasets': ['ellipses', 'lodopab']},
    'diptv': {
        'type': {'cls': 'DeepImagePriorCTReconstructor',
                 'module': 'dival.reconstructors.dip_ct_reconstructor'},
        'datasets': ['ellipses', 'lodopab']}
}
"""
Specification of reference configurations.

For each configuration key name a dict with the following fields is
specified:

    ``'type'`` : dict
        The reconstructor class, given by the following fields:

            ``'cls'`` : str
                The class name.
            ``'module'`` : str
                The module to import the class from.

    ``'datasets'`` : list of str
        List of standard dataset names the configuration is available for.
    ``'learned_params_spec'`` : dict, optional
        How learned parameters are stored.
        See also :meth:`LearnedReconstructor.save_learned_params` and
        :meth:`LearnedReconstructor.load_learned_params`.
        Valid fields are:

            ``'ext'`` : str, optional
                A single file with the given extension (e.g. ``'.pt'``).
                The param path (returned by :func:`get_params_path`) is
                suffixed by this.
            ``'dir'`` : ?, optional
                A directory.
                The param path (returned by :func:`get_params_path`) is equal
                to the directory path.
                *not implemented yet*
"""

DATASETS = ['ellipses', 'lodopab']
"""
List of standard datasets for which (some) reference reconstructor
configurations are available.
"""

DATA_URL = 'https://github.com/jleuschn/supp.dival/raw/master/reference_params'

def construct_reconstructor(reconstructor_key_name_or_type, dataset_name,
                            **kwargs):
    """
    Construct reference reconstructor object (not loading parameters).

    Note: see :func:get_reference_reconstructor to retrieve a reference
    reconstructor with optimized parameters.

    This function implements the constructors calls which are potentially
    specific to each configuration.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.
    kwargs : dict
        Keyword arguments.
        For CT configurations this includes the ``'impl'`` used by
        :class:`odl.tomo.RayTransform`.

    Raises
    ------
    ValueError
        If the configuration does not exist.
    NotImplementedError
        If construction is not implemented for the configuration.

    Returns
    -------
    reconstructor : :class:`Reconstructor`
        The reconstructor instance.
    """
    r_key_name, r_type = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    r_args = []
    r_kwargs = {}
    if dataset_name in ['ellipses', 'lodopab']:
        impl = kwargs.pop('impl', 'astra_cuda')
        dataset = get_standard_dataset(dataset_name, impl=impl)
        if r_key_name in ['fbp', 'fbpunet', 'iradonmap', 'learnedgd',
                          'learnedpd', 'tvadam', 'diptv']:
            ray_trafo = dataset.get_ray_trafo(impl=impl)
            r_args = [ray_trafo]
            r_kwargs['name'] = '{d}_{r}'.format(r=r_key_name, d=dataset_name)
        else:
            raise NotImplementedError(
                'reconstructor construction is not implemented for reference '
                'configuration \'{}\' for dataset \'{}\''
                .format(r_key_name, dataset_name))
    else:
        raise NotImplementedError(
            'reference reconstructor construction is not implemented for '
            'dataset \'{}\''.format(dataset_name))
    reconstructor = r_type(*r_args, **r_kwargs)
    return reconstructor

def validate_reconstructor_key_name_or_type(reconstructor_key_name_or_type,
                                            dataset_name):
    """
    Validate that a configuration exists and return both its key name
    and the reconstructor type.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.

    Raises
    ------
    ValueError
        If the configuration does not exist.

    Returns
    -------
    r_key_name : str
        Key name of the configuration.
    r_type : type
        Reconstructor type.
    """
    if isinstance(reconstructor_key_name_or_type, str):
        r_key_name = reconstructor_key_name_or_type
        if (r_key_name not in CONFIGURATIONS or
                dataset_name not in (
                CONFIGURATIONS[r_key_name]['datasets'])):
            raise ValueError('unknown reference configuration \'{}\' for '
                             'dataset \'{}\''.format(r_key_name, dataset_name))
        r_type = getattr(
            import_module(CONFIGURATIONS[r_key_name]['type']['module']),
            CONFIGURATIONS[r_key_name]['type']['cls'])
    else:
        r_type = reconstructor_key_name_or_type
        r_key_names = [k for k, v in CONFIGURATIONS.items()
                       if (v['type']['cls'] == r_type.__name__ and
                           dataset_name in v['datasets'])]
        if not r_key_names:
            raise ValueError('unknown reconstructor type {} for '
                             'dataset \'{}\''.format(r_type, dataset_name))
        r_key_name = r_key_names[0]
        if len(r_key_names) > 1:
            warn('There are multiple reference configurations for '
                 'reconstructor type {} and dataset \'{}\': {}. '
                 'Selecting \'{}\' now. To select another one, please specify '
                 'it by key name instead of reconstructor type.'
                 .format(r_type, dataset_name, r_key_names, r_key_name))
    return r_key_name, r_type

def get_params_path(reconstructor_key_name_or_type, dataset_name):
    """
    Return path of the parameters for a configuration.
    It can be passed to :class:`Reconstructor.load_params` as a single argument
    to load all parameters (hyper params and learned params).

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.

    Returns
    -------
    params_path : str
        Parameter path.
    """
    r_key_name, _ = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    params_path = os.path.join(DATA_PATH, dataset_name,
                               '{d}_{r}'.format(r=r_key_name, d=dataset_name))
    return params_path

def get_hyper_params_path(reconstructor_key_name_or_type, dataset_name):
    """
    Return path of the hyper parameters for a configuration.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.

    Returns
    -------
    hyper_params_path : str
        Hyper parameter path.
    """
    r_key_name, r_type = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    if (issubclass(r_type, LearnedReconstructor) and
            CONFIGURATIONS[r_key_name]['learned_params_spec'] == 'dir'):
        hyper_params_path = os.path.join(
            DATA_PATH, dataset_name,
            '{d}_{r}'.format(r=r_key_name, d=dataset_name),
            'hyper_params.json')
    else:  # learned parameters in single file or no learned parameters
        hyper_params_path = os.path.join(
            DATA_PATH, dataset_name,
            '{d}_{r}_hyper_params.json'.format(r=r_key_name, d=dataset_name))
    return hyper_params_path

def download_params(reconstructor_key_name_or_type, dataset_name,
                    include_learned=True):
    """
    Download parameters for a configuration.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.
    include_learned : bool, optional
        Whether to include learned parameters.
        Otherwise only hyper parameters are downloaded.
        Default: `True`.

    Raises
    ------
    NotImplementedError
        If trying to download learned parameters that are stored in a
        directory (instead of as a single file).
    ValueError
        If trying to download learned parameters for a configuration that
        does not specify how they are stored (as a single file or in a
        directory).
    """
    r_key_name, r_type = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    os.makedirs(DATA_PATH, exist_ok=True)
    params_path = get_params_path(r_key_name, dataset_name)
    hyper_params_url = ('{b}/{d}/{d}_{r}_hyper_params.json'
                        .format(b=DATA_URL, r=r_key_name, d=dataset_name))
    hyper_params_filename = params_path + '_hyper_params.json'
    os.makedirs(os.path.dirname(hyper_params_filename), exist_ok=True)
    with open(hyper_params_filename, 'wt') as file:
        r = requests.get(hyper_params_url)
        file.write(r.text)
    if include_learned and issubclass(r_type, LearnedReconstructor):
        learned_params_spec = CONFIGURATIONS[r_key_name]['learned_params_spec']
        if 'ext' in learned_params_spec:
            ext = learned_params_spec['ext']
            learned_params_url = (
                '{b}/{d}/{d}_{r}{e}'
                .format(b=DATA_URL, r=r_key_name, d=dataset_name, e=ext))
            learned_params_filename = params_path + ext
            download_file(learned_params_url, learned_params_filename,
                          md5sum=False)
        elif 'dir' in learned_params_spec:
            raise NotImplementedError(
                'automatic downloading of learned param directories is not '
                'implemented yet')
        else:
            raise ValueError('reference configuration \'{}\' misses '
                             'specification how learned params are stored'
                             .format(r_key_name))

def download_hyper_params(reconstructor_key_name_or_type, dataset_name):
    """
    Download hyper parameters for a configuration.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.
    """
    download_params(reconstructor_key_name_or_type, dataset_name,
                    include_learned=False)

def check_for_params(reconstructor_key_name_or_type, dataset_name,
                     include_learned=True, return_missing=False):
    """
    Return whether the parameter file(s) can be found.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.
    include_learned : bool, optional
        Whether to check for learned parameters, too.
        Default: `True`.
    return_missing : bool, optional
        Whether to return a list of missing files as second return value.
        Default: `False`.

    Raises
    ------
    NotImplementedError
        If trying to check for learned parameters that are stored in a
        directory (instead of as a single file).
    ValueError
        If trying to check for learned parameters for a configuration that
        does not specify how they are stored (as a single file or in a
        directory).

    Returns
    -------
    params_exist : bool
        Whether the parameter file(s) can be found.
    missing : list of str, optional
        List of missing files.
        Only returned if `return_missing=True`.
    """
    r_key_name, r_type = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    params_path = get_params_path(r_key_name, dataset_name)
    files = [params_path + '_hyper_params.json']
    if include_learned and issubclass(r_type, LearnedReconstructor):
        learned_params_spec = CONFIGURATIONS[r_key_name]['learned_params_spec']
        if 'ext' in learned_params_spec:
            ext = learned_params_spec['ext']
            files.append(params_path + ext)
        elif 'dir' in learned_params_spec:
            raise NotImplementedError(
                'checking for learned param directories is not implemented '
                'yet')
        else:
            raise ValueError('reference configuration \'{}\' misses '
                             'specification how learned params are stored'
                             .format(r_key_name))
    missing = [f for f in files if not os.path.isfile(f)]
    params_exist = not missing
    return (params_exist, missing) if return_missing else params_exist

def check_for_hyper_params(reconstructor_key_name_or_type, dataset_name):
    """
    Return whether the hyper parameter file can be found.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.

    Returns
    -------
    params_exist : bool
        Whether the hyper parameter file can be found.
    """
    params_exist = check_for_hyper_params(
        reconstructor_key_name_or_type, dataset_name,
        include_learned=False, return_missing=False)
    return params_exist

def get_reference_reconstructor(reconstructor_key_name_or_type, dataset_name,
                                pretrained=True, **kwargs):
    """
    Return a reference reconstructor.

    Parameters
    ----------
    reconstructor_key_name_or_type : str or type
        Key name of configuration or reconstructor type.
    dataset_name : str
        Standard dataset name.
    pretrained : bool, optional
        Whether learned parameters should be loaded (if any).
        Default: `True`.
    kwargs : dict
        Keyword arguments (passed to :func:`construct_reconstructor`).
        For CT configurations this includes the ``'impl'`` used by
        :class:`odl.tomo.RayTransform`.

    Raises
    ------
    RuntimeError
        If parameter files are missing and the user chooses not to download.

    Returns
    -------
    reconstructor : :class:`Reconstructor`
        The reference reconstructor.
    """
    r_key_name, r_type = validate_reconstructor_key_name_or_type(
        reconstructor_key_name_or_type, dataset_name)
    params_exist, missing = check_for_params(r_key_name, dataset_name,
        include_learned=pretrained, return_missing=True)
    if not params_exist:
        print("Reference configuration '{}' for dataset '{}' not found at the "
              "configured path '{}'. You can change this path with "
              "``dival.config.set_config('reference_params/datapath', ...)``."
              .format(r_key_name, dataset_name, DATA_PATH))
        print('Missing files are: {}.'.format(missing))
        print('Do you want to download it now? (y: download, n: cancel)')
        download = input_yes_no()
        if not download:
            raise RuntimeError('Reference configuration missing, cancelled')
        download_params(r_key_name, dataset_name)
    reconstructor = construct_reconstructor(r_key_name, dataset_name, **kwargs)
    params_path = get_params_path(r_key_name, dataset_name)
    reconstructor.load_hyper_params(params_path + '_hyper_params.json')
    if pretrained and issubclass(r_type, LearnedReconstructor):
        reconstructor.load_learned_params(params_path)
    return reconstructor
