# -*- coding: utf-8 -*-
"""
Provides the learned :class:`FBPUNetReconstructor` for the application of CT.
"""
import copy
from warnings import warn
from math import ceil
import numpy as np
from odl.tomo import fbp_op
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dival import Dataset, LearnedReconstructor
from skimage.measure import compare_psnr


def generate_fbp_cache(dataset, part, filename, ray_trafo, size=None):
    """
    Write filtered back-projections for a CT dataset part to file.

    Parameters
    ----------
    dataset : :class:`.Dataset`
        CT dataset from which the observations are used.
    part : {``'train'``, ``'validation'``, ``'test'``}
        The data part.
    filename : str
        The filename to store the FBP cache at (ending ``.npy``).
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform from which the FBP operator is constructed.
    size : int, optional
        Number of samples to use from the dataset.
        By default, all samples are used.
    """
    fbp = fbp_op(ray_trafo)
    num_samples = dataset.get_len(part=part) if size is None else size
    reco_fbps = np.empty((num_samples,) + dataset.shape[1], dtype=np.float32)
    obs = np.empty(dataset.shape[0], dtype=np.float32)
    tmp_fbp = fbp.range.element()
    for i in tqdm(range(num_samples), desc='generating FBP cache'):
        dataset.get_sample(i, part=part, out=(obs, False))
        fbp(obs, out=tmp_fbp)
        reco_fbps[i][:] = tmp_fbp
    np.save(filename, reco_fbps)


class CachedFBPDataset(Dataset):
    """Dataset combining the ground truth of a dataset with cached FBPs.

    Each sample is a pair of a FBP and a ground truth image.
    """
    def __init__(self, dataset, filenames):
        """
        Parameters
        ----------
        dataset : :class:`.Dataset`
            CT dataset from which the ground truth is used.
        filenames : dict
            Cached FBP filenames for the dataset parts.
            The part (``'train'``, ...) is the key to the dict.
            To generate the FBP files, :func:`generate_fbp_cache` can be used.
        """
        self.dataset = dataset
        self.filenames = filenames
        self.fbps = {}
        for part, filename in filenames.items():
            try:
                self.fbps[part] = np.load(filename, mmap_mode='r')
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Did not find FBP cache file '{}'. Please specify an "
                    'existing file or generate it using '
                    '`fbp_unet_reconstructor.generate_fbp_cache(...)`.'.format(
                        filename))
        self.train_len = (len(self.fbps['train']) if 'train' in self.fbps else
                          self.dataset.get_len('train'))
        self.validation_len = (len(self.fbps['validation']) if 'validation' in
                               self.fbps else
                               self.dataset.get_len('validation'))
        self.test_len = (len(self.fbps['test']) if 'test' in self.fbps else
                         self.dataset.get_len('test'))
        self.shape = (self.dataset.shape[1], self.dataset.shape[1])
        self.num_elements_per_sample = 2
        self.random_access = True
        super().__init__(space=(self.dataset.space[1], self.dataset.space[1]))

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        (_, gt) = self.dataset.get_sample(index, part=part,
                                          out=(False, out[1]))
        if isinstance(out[0], bool):
            fbp = self.fbps[part][index] if out[0] else None
        else:
            out[0][:] = self.fbps[part][index]
            fbp = out[0]
        return (fbp, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        (_, gt) = self.dataset.get_samples(key, part=part,
                                           out=(False, out[1]))
        if isinstance(out[0], bool):
            fbp = self.fbps[part][key] if out[0] else None
        else:
            out[0][:] = self.fbps[part][key]
            fbp = out[0]
        return (fbp, gt)


class FBPDataset(Dataset):
    """
    Dataset computing filtered back-projections for a CT dataset on the fly.

    Each sample is a pair of a FBP and a ground truth image.
    """
    def __init__(self, dataset, ray_trafo):
        """
        Parameters
        ----------
        dataset : :class:`.Dataset`
            CT dataset. FBPs are computed from the observations, the ground
            truth is taken directly from the dataset.
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        """
        self.dataset = dataset
        self.fbp_op = fbp_op(ray_trafo)
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.shape = (self.dataset.shape[1], self.dataset.shape[1])
        self.num_elements_per_sample = 2
        self.random_access = dataset.supports_random_access()
        super().__init__(space=(self.dataset.space[1], self.dataset.space[1]))

    def generator(self, part='train'):
        gen = self.dataset.generator(part=part)
        for (obs, gt) in gen:
            fbp = self.fbp_op(obs)
            yield (fbp, gt)

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        out_fbp = not (isinstance(out[0], bool) and not out[0])
        (obs, gt) = self.dataset.get_sample(index, part=part,
                                            out=(out_fbp, out[1]))
        if isinstance(out[0], bool):
            fbp = self.fbp_op(obs) if out[0] else None
        else:
            if out[0] in self.fbp_op.range:
                self.fbp_op(obs, out=out[0])
            else:
                out[0][:] = self.fbp_op(obs)
            fbp = out[0]
        return (fbp, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        out_fbp = not (isinstance(out[0], bool) and not out[0])
        (obs_arr, gt_arr) = self.dataset.get_samples(key, part=part,
                                                     out=(out_fbp, out[1]))
        if isinstance(out[0], bool) and out[0]:
            fbp_arr = np.empty((len(obs_arr),) + self.dataset.shape[1],
                               dtype=self.dataset.space[1].dtype)
        elif isinstance(out[0], bool) and not out[0]:
            fbp_arr = None
        else:
            fbp_arr = out[0]
        if out_fbp:
            tmp_fbp = self.fbp_op.range.element()
            for i in range(len(obs_arr)):
                self.fbp_op(obs_arr[i], out=tmp_fbp)
                fbp_arr[i][:] = tmp_fbp
        return (fbp_arr, gt_arr)


class FBPUNetReconstructor(LearnedReconstructor):
    HYPER_PARAMS = {
        'scales': {
            'default': 5,
            'retrain': True
        },
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'batch_size': {
            'default': 64,
            'retrain': True
        }
    }
    """
    CT Reconstructor applying filtered back-projection followed by a
    postprocessing U-Net (cf. [1]_).
    The U-Net architecture is similar to the one in [2]_.

    References
    ----------
    .. [1] K. H. Jin, M. T. McCann, E. Froustey, et al., 2017,
           "Deep Convolutional Neural Network for Inverse Problems in Imaging".
           IEEE Transactions on Image Processing.
           `doi:10.1109/TIP.2017.2713099
           <https://doi.org/10.1109/TIP.2017.2713099>`_
    .. [2] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018,
           "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           `doi:10.1109/CVPR.2018.00984
           <https://doi.org/10.1109/CVPR.2018.00984>`_
    """
    def __init__(self, ray_trafo, scales=None, epochs=None, batch_size=None,
                 num_data_loader_workers=8, use_cuda=True, show_pbar=True,
                 fbp_impl='astra_cuda', **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        scales : int, optional
            Number of scales in the U-Net (a hyper parameter).
        epochs : int, optional
            Number of epochs to train (a hyper parameter).
        batch_size : int, optional
            Batch size (a hyper parameter).
        num_data_loader_workers : int, optional
            Number of parallel workers to use for loading data.
        use_cuda : bool, optional
            Whether to use cuda for the U-Net.
        show_pbar : bool, optional
            Whether to show tqdm progress bars during the epochs.
        fbp_impl : str, optional
            The backend implementation passed to
            :class:`odl.tomo.RayTransform` in case no `ray_trafo` is specified.
            Then ``dataset.get_ray_trafo(impl=fbp_impl)`` is used to get the
            ray transform and FBP operator.
        """
        self.ray_trafo = ray_trafo
        self.fbp_op = fbp_op(self.ray_trafo)
        self.num_data_loader_workers = num_data_loader_workers
        self.use_cuda = use_cuda
        self.show_pbar = show_pbar
        self.fbp_impl = fbp_impl
        super().__init__(reco_space=self.ray_trafo.domain,
                         observation_space=self.ray_trafo.range, **kwargs)
        if epochs is not None:
            self.epochs = epochs
            if kwargs.get('hyper_params', {}).get('epochs') is not None:
                warn("hyper parameter 'epochs' overridden by constructor argument")
        if batch_size is not None:
            self.batch_size = batch_size
            if kwargs.get('hyper_params', {}).get('batch_size') is not None:
                warn("hyper parameter 'batch_size' overridden by constructor argument")
        if scales is not None:
            self.scales = scales
            if kwargs.get('hyper_params', {}).get('scales') is not None:
                warn("hyper parameter 'scales' overridden by constructor argument")
        
    def get_epochs(self):
        return self.hyper_params['epochs']
        
    def set_epochs(self, epochs):
        self.hyper_params['epochs'] = epochs

    epochs = property(get_epochs, set_epochs)
        
    def get_batch_size(self):
        return self.hyper_params['batch_size']
        
    def set_batch_size(self, batch_size):
        self.hyper_params['batch_size'] = batch_size

    batch_size = property(get_batch_size, set_batch_size)
        
    def get_scales(self):
        return self.hyper_params['scales']
        
    def set_scales(self, scales):
        self.hyper_params['scales'] = scales

    scales = property(get_scales, set_scales)

    def train(self, dataset):
        try:
            self.fbp_dataset = dataset.fbp_dataset
        except AttributeError:
            self.fbp_dataset = FBPDataset(dataset, self.ray_trafo)

        num_workers = self.num_data_loader_workers
        if not self.fbp_dataset.supports_random_access():
            warn('Dataset does not support random access. Shuffling will not '
                 'work, and only 1 worker will be used for data loading.')
            num_workers = 1
        fbp_dataset_train = self.fbp_dataset.create_torch_dataset(
            part='train', reshape=((1,) + self.fbp_dataset.shape[0],
                                   (1,) + self.fbp_dataset.shape[1]))
        fbp_dataset_validation = self.fbp_dataset.create_torch_dataset(
            part='validation', reshape=((1,) + self.fbp_dataset.shape[0],
                                        (1,) + self.fbp_dataset.shape[1]))
        
        ttype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.net = get_skip_model(scales=self.scales).type(ttype)
        self.net = nn.DataParallel(self.net)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                    gamma=0.1)

        dataloaders = {'train': DataLoader(
                           fbp_dataset_train, batch_size=self.batch_size,
                           num_workers=num_workers, shuffle=True,
                           pin_memory=True),
                       'validation': DataLoader(
                           fbp_dataset_validation, batch_size=self.batch_size,
                           num_workers=num_workers,
                           shuffle=True, pin_memory=True)}

        dataset_sizes = {'train': len(fbp_dataset_train),
                         'validation': len(fbp_dataset_validation)}

        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_psnr = 0

        device = (torch.device('cuda:0') if self.use_cuda else
                  torch.device('cpu'))
        self.net.to(device)
        self.net.train()

        self.history = {'train': {'loss': [],
                                  'psnr': []},
                        'validation': {'loss': [],
                                       'psnr': []}}

        for epoch in range(self.epochs):
            print('epoch {:d}/{:d}'.format(epoch, self.epochs), flush=True)
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.net.train()  # Set model to training mode
                else:
                    self.net.eval()   # Set model to evaluate mode

                running_psnr = 0.0
                # Iterate over data.
                batch_id = 0
                with tqdm(dataloaders[phase], desc='epoch {:d}'.format(epoch),
                          disable=not self.show_pbar) as pbar:
                    for inputs, labels in pbar:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.net(inputs)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        current_loss = loss.item()
                        current_psnr = 0.0
                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            current_psnr += compare_psnr(
                                outputs_, labels_, data_range=np.max(labels_))
                        current_psnr /= outputs.shape[0]
                        running_psnr += current_psnr
                        self.history[phase]['loss'].append(current_loss)
                        self.history[phase]['psnr'].append(current_psnr)
                        pbar.set_postfix({'phase': phase,
                                          'loss': current_loss,
                                          'psnr': current_psnr})

                        batch_id += 1

                    num_steps = ceil(dataset_sizes[phase] / self.batch_size)
                    epoch_psnr = running_psnr / num_steps

                    # deep copy the model
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = copy.deepcopy(self.net.state_dict())
                scheduler.step(epoch=epoch)

        print('Best val psnr: {:4f}'.format(best_psnr))

        self.net.load_state_dict(best_model_wts)

    def _reconstruct(self, observation):
        self.net.eval()
        fbp = self.fbp_op(observation)
        ttype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        fbp_tensor = (torch.from_numpy(np.asarray(fbp)[None, None])
                      .type(ttype))
        reco_tensor = self.net(fbp_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)

    def save_params(self, filename):
        """
        Save U-Net state to file using :meth:`torch.save`.

        Parameters
        ----------
        filename : str
            Filename (ending ``.pt``).
        """
        torch.save(self.net.state_dict(), filename)

    def load_params(self, filename):
        """
        Load U-Net state from file using :meth:`torch.load`.

        Parameters
        ----------
        filename : str
            Filename (ending ``.pt``).
        """
        ttype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.net = get_skip_model(scales=self.scales).type(ttype)
        self.net = nn.DataParallel(self.net)
        map_location = 'cuda:0' if self.use_cuda else 'cpu'
        state_dict = torch.load(filename, map_location=map_location)
        # backwards-compatibility with non-data_parallel weights
        data_parallel = list(state_dict.keys())[0].startswith('module.')
        if not data_parallel:
            state_dict = {('module.' + k): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)


def get_skip_model(scales=5, skip=4):
    assert(1 <= scales <= 6)
    channels = (16, 32, 64, 64, 128, 128)
    skip_channels = [skip] * (scales)
    return Skip(in_ch=1, out_ch=1, channels=channels[:scales],
                skip_channels=skip_channels)


class Skip(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels):
        super(Skip, self).__init__()
        assert(len(channels) == len(skip_channels))
        self.scales = len(channels)

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0])
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i-1],
                                       out_ch=channels[i]))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i], out_ch=channels[-i-1],
                                   skip_ch=skip_channels[-i]))

        self.outc = OutBlock(in_ch=channels[0], out_ch=out_ch)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales-1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales-1):
            x = self.up[i](x, xs[-2-i])

        return torch.sigmoid(self.outc(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=to_pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=to_pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=to_pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch + skip_ch),
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                      padding=to_pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=to_pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(skip_ch),
            nn.LeakyReLU(0.2, inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
