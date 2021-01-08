# -*- coding: utf-8 -*-
"""
This script resimulates observation data from the ground truth of the
`LoDoPaB-CT dataset <https://doi.org/10.5281/zenodo.3384092>`_ with a different
setting:

    * A fan beam (2d cone beam) geometry is used instead of 2d parallel beam.
    * 5% Gaussian noise is added to the post-log observations instead of
      applying Poisson noise to the pre-log observations.

Prerequisites:

    * imported libraries + `astra-toolbox`
    * unzipped LoDoPaB-CT dataset stored in `DATA_PATH`, at least the ground
      truth

Notes and limitations:

    * At the corners of the image, stronger artifacts are visible in the FBPs.
      This is a consequence of two factors:

          1. the fan beam geometry chosen for this resimulation
          2. the center crop to 362x362 pixels that has been applied to the
             ground truth images in LoDoPaB-CT (which was done because most
             images only contain valid values in a circle of diameter 362px)

      These artifacts will be less significant in a more realistic scenario
      with a non-cropped field-of-view, as the patient's body will not expand
      to the edges.
      This should be considered when evaluating the performance of
      reconstruction methods on the simulations created by this script.
"""
import os
from itertools import islice
from math import ceil
import numpy as np
import odl
from tqdm import tqdm
from skimage.transform import resize
import h5py

from dival.datasets.dataset import Dataset, GroundTruthDataset
from dival.datasets.lodopab_dataset import NUM_SAMPLES_PER_FILE, LEN
from dival.config import get_config


# path to lodopab dataset (input and output: ground truth data is read from
# this path and resimulated observations are written to this path)
DATA_PATH = get_config('lodopab_dataset/data_path')
# DATA_PATH = '/localdata/lodopab'  # or specify path explicitly

# name for the resimulated observations, the output HDF5 files will be named
# e.g. '{OBSERVATION_NAME}_train_000.hdf5'
OBSERVATION_NAME = 'observation_fan_beam'

# original ground truth and reconstruction image shape
RECO_IM_SHAPE = (362, 362)
# image shape for simulation (in order to avoid inverse crime)
IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)

# ~26cm x 26cm images
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]
# note: these values are used to define the domain on which the ODL
# RayTransform operates and thereby influence the scaling of the simulated
# observation values

# fan beam (2d cone beam) geometry parameters
SRC_RADIUS = 0.541  # 541 mm,
DET_RADIUS = .949075012 - SRC_RADIUS  # ~408 mm
# these values were chosen because they occur multiple times in DICOM fields
# (0018, 1110) Distance Source to Detector
# (0018, 1111) Distance Source to Patient
NUM_ANGLES = 1000

# ODL RayTransform backend
IMPL = 'astra_cuda'  # alternative: 'astra_cpu'


class LoDoPaBGroundTruthDataset(GroundTruthDataset):
    """
    Dataset providing only the ground truth data of the LoDoPaB-CT dataset.

    The images are (very slightly processed) reconstruction slices from the
    `LIDC-IDRI
    <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_ dataset.
    """

    def __init__(self):
        self.shape = RECO_IM_SHAPE
        self.train_len = LEN['train']
        self.validation_len = LEN['validation']
        self.test_len = LEN['test']
        self.random_access = False
        space = odl.uniform_discr(MIN_PT, MAX_PT, self.shape, dtype=np.float32)
        super().__init__(space=space)

    def generator(self, part='train'):
        num_files = ceil(self.get_len(part) / NUM_SAMPLES_PER_FILE)
        for i in range(num_files):
            with h5py.File(
                    os.path.join(DATA_PATH,
                                 'ground_truth_{}_{:03d}.hdf5'
                                 .format(part, i)), 'r') as file:
                ground_truth_data = file['data'][:]
            num_samples_in_file = (
                min((i+1) * NUM_SAMPLES_PER_FILE, LEN[part])
                - i * NUM_SAMPLES_PER_FILE)  # calculate number of samples
            # in file because the last ground truth file contains less than 128
            # valid samples while file['data'].shape still is (128, 362, 362)
            for gt_arr in islice(ground_truth_data, num_samples_in_file):
                yield self.space.element(gt_arr)


gt_dataset = LoDoPaBGroundTruthDataset()

# image space for reconstruction
reco_space = gt_dataset.space
# image space for simulation with different resolution to avoid inverse crime
space = odl.uniform_discr(min_pt=reco_space.min_pt,
                          max_pt=reco_space.max_pt,
                          shape=IM_SHAPE, dtype=np.float32)

# define fan beam (2d cone beam) geometry for reconstruction and simulation
reco_geometry = odl.tomo.cone_beam_geometry(
    reco_space, SRC_RADIUS, DET_RADIUS, NUM_ANGLES)
geometry = odl.tomo.cone_beam_geometry(
    space, SRC_RADIUS, DET_RADIUS, NUM_ANGLES,
    det_shape=reco_geometry.detector.shape)  # use the number of detector
                                             # pixels that ODL auto-computed
                                             # for reco_geometry

# create noise-less forward operator for simulation
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

class _ResizeOperator(odl.Operator):
    def __init__(self):
        super().__init__(reco_space, space)

    def _call(self, x, out):
        out.assign(space.element(resize(x, IM_SHAPE, order=1)))

resize_op = _ResizeOperator()
forward_op = ray_trafo * resize_op  # resize first, then apply ray transform

# create observation ground truth pair dataset that simulates on the fly
dataset = gt_dataset.create_pair_dataset(
    forward_op=forward_op, noise_type='white',
    noise_kwargs={'relative_stddev': True,
                  'stddev': 0.05},  # 5% gaussian noise
    noise_seeds={'train': 4, 'validation': 5, 'test': 6})

# create noise-less forward operator for reconstruction
def get_reco_ray_trafo(**kwargs):
    return odl.tomo.RayTransform(reco_space, reco_geometry, **kwargs)
reco_ray_trafo = get_reco_ray_trafo(impl=IMPL)

# provide RayTransform getter and instance as attributes of the dataset (like
# for the CT datasets returned by get_standard_dataset)
dataset.get_ray_trafo = get_reco_ray_trafo
dataset.ray_trafo = reco_ray_trafo

# The dataset can already be used at this point. It provides a generator for
# each part that simulates the observations on the fly. There are however some
# limitations, which can be overcome by storing the observations:
#     * Random access (by index) to the samples is not possible.
#       This is a general limitation of the generator-based implementation in
#       `dival.datasets.dataset.ObservationGroundTruthPairDataset`.
#       Since the samples are shuffled already, this would be fine for
#       training a network, but random access is desirable for other purposes.
#     * Simulating on the fly might be too slow.
#     * Potential irreproducibility, e.g. due to changes in used libraries.

# %% optional: plot first three train images and fbp reconstructions
import matplotlib.pyplot as plt
from dival.util.plot import plot_images
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
fbp_reconstructor = FBPReconstructor(dataset.ray_trafo,
                                     hyper_params={'filter_type': 'Hann',
                                                   'frequency_scaling': 1.0})
for i, (obs, gt) in islice(enumerate(dataset.generator(part='train')), 3):
    reco = fbp_reconstructor.reconstruct(obs)
    reco = np.clip(reco, 0., 1.)
    _, ax = plot_images([reco, gt], fig_size=(10, 4))
    ax[0].set_title('FBP reconstruction')
    ax[1].set_title('Ground truth')
    psnr = PSNR(reco, gt)
    ssim = SSIM(reco, gt)
    ax[0].set_xlabel('PSNR: {:.2f}dB, SSIM: {:.3f}'.format(psnr, ssim))
    print('metrics for FBP reconstruction on sample {:d}:'.format(i))
    print('PSNR: {:.2f}dB, SSIM: {:.3f}'.format(psnr, ssim))
plt.show()

# %% simulate and store fan beam observations

from dival.util.input import input_yes_no
print('start simulating and storing fan beam observations for all lodopab '
      'ground truth samples? [y]/n')
input_yes_no()

obs_shape = dataset.ray_trafo.range.shape
for part in ['train', 'validation', 'test']:
    for i, (obs, gt) in enumerate(tqdm(
            dataset.generator(part),
            desc='simulating part \'{}\''.format(part),
            total=dataset.get_len(part))):
        filenumber = i // NUM_SAMPLES_PER_FILE
        idx_in_file = i % NUM_SAMPLES_PER_FILE
        obs_filename = os.path.join(
            DATA_PATH,
            '{}_{}_{:03d}.hdf5'.format(OBSERVATION_NAME, part, filenumber))
        with h5py.File(obs_filename, 'a') as observation_file:
            observation_dataset = observation_file.require_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + obs_shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + obs_shape,
                dtype=np.float32, exact=True, fillvalue=np.nan, chunks=True)
            observation_dataset[idx_in_file] = obs

            # resize last file after storing last sample
            if i == dataset.get_len(part) - 1:
                n_files = ceil(dataset.get_len(part) / NUM_SAMPLES_PER_FILE)
                observation_dataset.resize(
                    dataset.get_len(part) - (n_files-1) * NUM_SAMPLES_PER_FILE,
                    axis=0)

# %% class for accessing the stored dataset
class LoDoPaBFanBeamDataset(Dataset):
    """
    CT Dataset using the ground truth data from the LoDoPaB-CT dataset and
    observations simulated for a fan beam geometry.
    5% Gaussian noise is added to the observations to simulate a lower dose.

    The ground truth images from LoDoPaB-CT are (very slightly processed)
    reconstruction slices from the `LIDC-IDRI
    <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_ dataset.
    """
    def __init__(self, impl='astra_cuda'):
        """
        Parameters
        ----------
        impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``}, optional
            Implementation passed to :class:`odl.tomo.RayTransform` to
            construct :attr:`ray_trafo`.
        """
        self.shape = (dataset.ray_trafo.range.shape, (362, 362))
        self.num_elements_per_sample = 2
        self.train_len = LEN['train']
        self.validation_len = LEN['validation']
        self.test_len = LEN['test']
        self.random_access = True

        self.geometry = reco_geometry
        super().__init__(
            space=(dataset.ray_trafo.range, dataset.ray_trafo.domain))
        self.ray_trafo = self.get_ray_trafo(impl=impl)

    def generator(self, part='train'):
        num_files = ceil(self.get_len(part) / NUM_SAMPLES_PER_FILE)
        observation_trafo = self.__get_observation_trafo()
        for i in range(num_files):
            with h5py.File(
                    os.path.join(DATA_PATH,
                                 'ground_truth_{}_{:03d}.hdf5'
                                 .format(part, i)), 'r') as file:
                ground_truth_data = file['data'][:]
            with h5py.File(
                    os.path.join(DATA_PATH,
                                 '{}_{}_{:03d}.hdf5'
                                 .format(OBSERVATION_NAME, part, i)),
                    'r') as file:
                observation_data = file['data'][:]
            for gt_arr, obs_arr in zip(ground_truth_data, observation_data):
                ground_truth = self.space[1].element(gt_arr)
                observation = self.space[0].element(obs_arr)
                observation_trafo(observation)

                yield (observation, ground_truth)

    def get_ray_trafo(self, **kwargs):
        return odl.tomo.RayTransform(self.space[1], self.geometry, **kwargs)

    def get_sample(self, index, part='train', out=None):
        len_part = self.get_len(part)
        if index >= len_part or index < -len_part:
            raise IndexError("index {} out of bounds for part '{}' ({:d})"
                             .format(index, part, len_part))
        if index < 0:
            index += len_part
        if out is None:
            out = (True, True)
        (out_observation, out_ground_truth) = out
        if self.sorted_by_patient:
            index = self._idx_sorted_by_patient[part][index]
        file_index = index // NUM_SAMPLES_PER_FILE
        index_in_file = index % NUM_SAMPLES_PER_FILE
        if isinstance(out_observation, bool):
            obs = self.space[0].zero() if out_observation else None
        else:
            obs = out_observation
        if isinstance(out_ground_truth, bool):
            gt = self.space[1].zero() if out_ground_truth else None
        else:
            gt = out_ground_truth
        if obs is not None:
            with h5py.File(
                    os.path.join(DATA_PATH,
                                 '{}_{}_{:03d}.hdf5'
                                 .format(OBSERVATION_NAME, part, file_index)),
                    'r') as file:
                file['data'].read_direct(np.asarray(obs)[np.newaxis],
                                         np.s_[index_in_file:index_in_file+1],
                                         np.s_[0:1])
            observation_trafo = self.__get_observation_trafo()
            observation_trafo(obs)
        if gt is not None:
            with h5py.File(
                    os.path.join(DATA_PATH,
                                 'ground_truth_{}_{:03d}.hdf5'
                                 .format(part, file_index)), 'r') as file:
                file['data'].read_direct(np.asarray(gt)[np.newaxis],
                                         np.s_[index_in_file:index_in_file+1],
                                         np.s_[0:1])
        return (obs, gt)

    def get_samples(self, key, part='train', out=None):
        len_part = self.get_len(part)
        if isinstance(key, slice):
            key_start = (0 if key.start is None else
                         (key.start if key.start >= 0 else
                          max(0, len_part+key.start)))
            key_stop = (len_part if key.stop is None else
                        (key.stop if key.stop >= 0 else
                         max(0, len_part+key.stop)))
            range_ = range(key_start, key_stop, key.step or 1)
        elif isinstance(key, range):
            range_ = key
        else:
            raise TypeError('`key` expected to have type `slice` or `range`')
        if range_.step < 0:
            raise ValueError('key {} invalid, negative steps are not '
                             'implemented yet'.format(key))
        if range_[-1] >= len_part:
            raise IndexError("key {} out of bounds for part '{}' ({:d})"
                             .format(key, part, len_part))
        range_files = range(range_[0] // NUM_SAMPLES_PER_FILE,
                            range_[-1] // NUM_SAMPLES_PER_FILE + 1)
        if out is None:
            out = (True, True)
        (out_observation, out_ground_truth) = out
        # compute slice objects
        slices_files = []
        slices_data = []
        data_count = 0
        for i in range_files:
            if i == range_files.start:
                start = range_.start % NUM_SAMPLES_PER_FILE
            else:
                start = (range_.start - i*NUM_SAMPLES_PER_FILE) % range_.step
            if i == range_files[-1]:
                stop = range_[-1] % NUM_SAMPLES_PER_FILE + 1
            else:
                __next_start = ((range_.start - (i+1)*NUM_SAMPLES_PER_FILE)
                                % range_.step)
                stop = (__next_start - range_.step) % NUM_SAMPLES_PER_FILE + 1
            s = slice(start, stop, range_.step)
            slices_files.append(s)
            len_slice = ceil((s.stop-s.start) / s.step)
            slices_data.append(slice(data_count, data_count+len_slice))
            data_count += len_slice
        # read data
        if isinstance(out_observation, bool):
            obs_arr = (np.empty((len(range_),) + self.shape[0],
                                dtype=np.float32) if out_observation else None)
        else:
            obs_arr = out_observation
        if isinstance(out_ground_truth, bool):
            gt_arr = (np.empty((len(range_),) + self.shape[1],
                               dtype=np.float32) if out_ground_truth else None)
        else:
            gt_arr = out_ground_truth
        if obs_arr is not None:
            for i, slc_f, slc_d in zip(range_files, slices_files, slices_data):
                with h5py.File(
                        os.path.join(DATA_PATH,
                                     '{}_{}_{:03d}.hdf5'
                                     .format(OBSERVATION_NAME, part, i)),
                        'r') as file:
                    file['data'].read_direct(obs_arr, slc_f, slc_d)
        if gt_arr is not None:
            for i, slc_f, slc_d in zip(range_files, slices_files, slices_data):
                with h5py.File(
                        os.path.join(DATA_PATH,
                                     'ground_truth_{}_{:03d}.hdf5'
                                     .format(part, i)), 'r') as file:
                    file['data'].read_direct(gt_arr, slc_f, slc_d)
        return (obs_arr, gt_arr)


# %% alternative simulation using multiprocessing
# Below is unfinished code for a parallelized simulation, which could be very
# beneficial if the RayTransform backend IMPL='astra_cpu' is chosen above.
# Similar code was used for simulating the standard parallel beam observations.
# See footnote 8 in https://arxiv.org/pdf/1910.01113.pdf for a reason why
# 'astra_cpu' was chosen there.
# The following pickle error occurs:
#     AttributeError: Can't pickle local object
#     'FanBeamGeometry.__init__.<locals>.<lambda>'


# import multiprocessing
# from dival.util.odl_utility import apply_noise

# for part in ['train', 'validation', 'test']:
#     rs = np.random.RandomState({'train': 4, 'validation': 5, 'test': 6}[part])
#     # initialize with fixed seed per part
#     gen = gt_dataset.generator(part)
#     n_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
#     for filenumber in tqdm(range(n_files), desc=part):
#         obs_filename = os.path.join(
#             DATA_PATH,
#             '{}_{}_{:03d}.hdf5'.format(OBSERVATION_NAME, part, filenumber))
#         with h5py.File(obs_filename, 'w') as observation_file:
#             observation_dataset = observation_file.create_dataset(
#                 'data',
#                 shape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
#                 maxshape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
#                 dtype=np.float32, chunks=True)
#             im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]
#             with multiprocessing.Pool(20) as pool:
#                 data_buf = pool.map(forward_op, im_buf)

#             for i, (im, data) in enumerate(zip(im_buf, data_buf)):
#                 apply_noise(data,
#                             noise_type=dataset.noise_type,
#                             noise_kwargs=dataset.noise_kwargs,
#                             random_state=rs)
#                 observation_dataset[i] = data

#             # resize last file
#             if filenumber == n_files - 1:
#                 observation_dataset.resize(
#                     LEN[part] - (n_files - 1) * NUM_SAMPLES_PER_FILE,
#                     axis=0)
