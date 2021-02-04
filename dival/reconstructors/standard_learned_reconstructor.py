# -*- coding: utf-8 -*-
try:
    import torch
except ImportError:
    raise ImportError('missing PyTorch')

import os
from copy import deepcopy
from math import ceil
import odl
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    TENSORBOARD_AVAILABLE = False
else:
    TENSORBOARD_AVAILABLE = True
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from dival.reconstructors import LearnedReconstructor
from dival.measure import PSNR
from dival.util.torch_utility import load_state_dict_convert_data_parallel


class StandardLearnedReconstructor(LearnedReconstructor):
    """
    Standard learned reconstructor base class.

    Provides a default implementation that only requires subclasses to
    implement :meth:`init_model`.

    By default, the Adam optimizer is used. This can be changed by
    reimplementing :meth:`init_optimizer`.
    Also, a OneCycleLR scheduler is used by default, which can be changed by
    reimplementing :meth:`init_scheduler`.

    The training implementation selects the best model reached after an integer
    number of epochs based on the validation set.

    The hyper parameter ``'normalize_by_opnorm'`` selects whether
    :attr:`op` should be normalized by the operator norm.
    In this case, the inputs to :attr:`model` are divided by the operator norm.

    Attributes
    ----------
    model : :class:`torch.nn.Module` or `None`
        The neural network.
        Must be initialized by the subclass :meth:`init_model` implementation.
    non_normed_op : :class:`odl.operator.Operator`
        The original `op` passed to :meth:`__init__`, regardless of
        ``self.hyper_params['normalize_by_opnorm']``.
        See also :attr:`op`.
    """

    HYPER_PARAMS = deepcopy(LearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'batch_size': {
            'default': 64,
            'retrain': True
        },
        'lr': {
            'default': 0.01,
            'retrain': True
        },
        'normalize_by_opnorm': {
            'default': False,
            'retrain': True
        }
    })

    def __init__(self, op, hyper_params=None, num_data_loader_workers=8,
                 use_cuda=True, show_pbar=True, log_dir=None,
                 log_num_validation_samples=0,
                 save_best_learned_params_path=None, torch_manual_seed=1,
                 shuffle='auto', worker_init_fn=None, **kwargs):
        """
        Parameters
        ----------
        op : :class:`odl.operator.Operator`
            Forward operator.
        num_data_loader_workers : int, optional
            Number of parallel workers to use for loading data.
        use_cuda : bool, optional
            Whether to use cuda for the U-Net.
        show_pbar : bool, optional
            Whether to show tqdm progress bars during the epochs.
        log_dir : str, optional
            Tensorboard log directory (name of sub-directory in utils/logs).
            If `None`, no logs are written.
        log_num_valiation_samples : int, optional
            Number of validation images to store in tensorboard logs.
            This option only takes effect if ``log_dir is not None``.
        save_best_learned_params_path : str, optional
            Save best model weights during training under the specified path by
            calling :meth:`save_learned_params`.
        torch_manual_seed : int, optional
            Fixed seed to set by ``torch.manual_seed`` before training.
            The default is `1`. It can be set to `None` or `False` to disable
            the manual seed.
        shuffle : {``'auto'``, ``False``, ``True``}, optional
            Whether to use shuffling when loading data.
            When ``'auto'`` is specified (the default), ``True`` is used iff
            the dataset passed to :meth:`train` supports random access.
        worker_init_fn : callable, optional
            Callable `worker_init_fn` passed to
            :meth:`torch.utils.data.DataLoader.__init__`, which can be used to
            configure the dataset copies for different worker instances
            (cf. `torch's IterableDataset docs <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`_)
        """
        super().__init__(reco_space=op.domain,
                         observation_space=op.range,
                         hyper_params=hyper_params, **kwargs)
        self.non_normed_op = op
        self.num_data_loader_workers = num_data_loader_workers
        self.use_cuda = use_cuda
        self.show_pbar = show_pbar
        self.log_dir = log_dir
        self.log_num_validation_samples = log_num_validation_samples
        self.save_best_learned_params_path = save_best_learned_params_path
        self.torch_manual_seed = torch_manual_seed
        self.shuffle = shuffle
        self.worker_init_fn = worker_init_fn
        self.model = None
        self._optimizer = None
        self._scheduler = None

        self._opnorm = None

        self.device = (torch.device('cuda:0')
                       if self.use_cuda and torch.cuda.is_available() else
                       torch.device('cpu'))

    @property
    def opnorm(self):
        if self._opnorm is None:
            self._opnorm = odl.power_method_opnorm(self.non_normed_op)
        return self._opnorm

    @property
    def op(self):
        """
        :class:`odl.operator.Operator` :
        The forward operator, normalized if
        ``self.hyper_params['normalize_by_opnorm']`` is ``True``.
        """
        if self.normalize_by_opnorm:
            return (1./self.opnorm) * self.non_normed_op
        return self.non_normed_op

    def eval(self, test_data):
        self.model.eval()

        running_psnr = 0.0
        with tqdm(test_data, desc='test ',
                  disable=not self.show_pbar) as pbar:
            for obs, gt in pbar:
                rec = self.reconstruct(obs)
                running_psnr += PSNR(rec, gt)

        return running_psnr / len(test_data)

    def train(self, dataset):
        if self.torch_manual_seed:
            torch.random.manual_seed(self.torch_manual_seed)

        self.init_transform(dataset=dataset)

        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape),
            transform=self._transform)

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        # reset model before training
        self.init_model()

        criterion = torch.nn.MSELoss()
        self.init_optimizer(dataset_train=dataset_train)

        # create PyTorch dataloaders
        shuffle = (dataset.supports_random_access() if self.shuffle == 'auto'
                   else self.shuffle)
        data_loaders = {
            'train': DataLoader(
                dataset_train, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers, shuffle=shuffle,
                pin_memory=True, worker_init_fn=self.worker_init_fn),
            'validation': DataLoader(
                dataset_validation, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers, shuffle=shuffle,
                pin_memory=True, worker_init_fn=self.worker_init_fn)}

        dataset_sizes = {'train': len(dataset_train),
                         'validation': len(dataset_validation)}

        self.init_scheduler(dataset_train=dataset_train)
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))

        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = 0

        if self.log_dir is not None:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError(
                    'Missing tensorboard. Please install it or disable '
                    'logging by specifying `log_dir=None`.')
            writer = SummaryWriter(log_dir=self.log_dir, max_queue=0)
            validation_samples = dataset.get_data_pairs(
                'validation', self.log_num_validation_samples)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1),
                          disable=not self.show_pbar) as pbar:
                    for inputs, labels in pbar:
                        if self.normalize_by_opnorm:
                            inputs = (1./self.opnorm) * inputs
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()
                                if (self._scheduler is not None and
                                        schedule_every_batch):
                                    self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, labels_)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})
                        if self.log_dir is not None and phase == 'train':
                            step = (epoch * ceil(dataset_sizes['train']
                                                 / self.batch_size)
                                    + ceil(running_size / self.batch_size))
                            writer.add_scalar(
                                'loss/{}'.format(phase),
                                torch.tensor(running_loss/running_size), step)
                            writer.add_scalar(
                                'psnr/{}'.format(phase),
                                torch.tensor(running_psnr/running_size), step)

                    if (self._scheduler is not None
                            and not schedule_every_batch):
                        self._scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if self.log_dir is not None and phase == 'validation':
                        step = (epoch+1) * ceil(dataset_sizes['train']
                                                / self.batch_size)
                        writer.add_scalar('loss/{}'.format(phase),
                                          epoch_loss, step)
                        writer.add_scalar('psnr/{}'.format(phase),
                                          epoch_psnr, step)

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if self.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.save_best_learned_params_path)

                if (phase == 'validation' and self.log_dir is not None and
                        self.log_num_validation_samples > 0):
                    with torch.no_grad():
                        val_images = []
                        for (y, x) in validation_samples:
                            y = torch.from_numpy(
                                np.asarray(y))[None, None].to(self.device)
                            x = torch.from_numpy(
                                np.asarray(x))[None, None].to(self.device)
                            reco = self.model(y)
                            reco -= torch.min(reco)
                            reco /= torch.max(reco)
                            val_images += [reco, x]
                        writer.add_images(
                            'validation_samples', torch.cat(val_images),
                            (epoch + 1) * (ceil(dataset_sizes['train'] /
                                                self.batch_size)),
                            dataformats='NCWH')

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)

    def init_transform(self, dataset):
        """
        Initialize the transform (:attr:`_transform`) that is applied on each
        training sample, e.g. for data augmentation.
        In the default implementation of :meth:`train`, it is passed to
        :meth:`Dataset.create_torch_dataset` when creating the training (but
        not the validation) torch dataset, which applies the transform to the
        (tuple of) torch tensor(s) right before returning, i.e. after reshaping
        to ``(1,) + orig_shape``.

        The default implementation of this method disables the transform by
        assigning `None`.
        Called in :meth:`train` at the beginning, i.e. before calling
        :meth:`init_model`, :meth:`init_optimizer` and :meth:`init_scheduler`.

        Parameters
        ----------
        dataset : :class:`dival.datasets.dataset.Dataset`
            The dival dataset passed to :meth:`train`.
        """
        self._transform = None

    @property
    def transform(self):
        """
        callable :
        Transform that is applied on each sample, usually set by
        :meth:`init_transform`, which gets called in :meth:`train`.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value

    def init_model(self):
        """
        Initialize :attr:`model`.
        Called in :meth:`train` after calling :meth:`init_transform`, but
        before calling :meth:`init_optimizer` and :meth:`init_scheduler`.
        """
        raise NotImplementedError

    def init_optimizer(self, dataset_train):
        """
        Initialize the optimizer.
        Called in :meth:`train`, after calling :meth:`init_transform` and
        :meth:`init_model`, but before calling :meth:`init_scheduler`.

        Parameters
        ----------
        dataset_train : :class:`torch.utils.data.Dataset`
            The training (torch) dataset constructed in :meth:`train`.
        """
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def init_scheduler(self, dataset_train):
        """
        Initialize the learning rate scheduler.
        Called in :meth:`train`, after calling :meth:`init_transform`,
        :meth:`init_model` and :meth:`init_optimizer`.

        Parameters
        ----------
        dataset_train : :class:`torch.utils.data.Dataset`
            The training (torch) dataset constructed in :meth:`train`.
        """
        self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self._optimizer, max_lr=self.lr,
            steps_per_epoch=ceil(len(dataset_train) / self.batch_size),
            epochs=self.epochs)

    @property
    def scheduler(self):
        """
        torch learning rate scheduler :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

    def _reconstruct(self, observation):
        self.model.eval()
        with torch.set_grad_enabled(False):
            obs_tensor = torch.from_numpy(
                np.asarray(observation)[None, None])
            if self.normalize_by_opnorm:
                obs_tensor = obs_tensor / self.opnorm
            obs_tensor = obs_tensor.to(self.device)
            reco_tensor = self.model(obs_tensor)
            reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)

    def save_learned_params(self, path):
        path = path if path.endswith('.pt') else path + '.pt'
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_learned_params(self, path, convert_data_parallel='auto'):
        """Load learned parameters from file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters are stored.
            Implementations may interpret this as a file path or as a directory
            path for multiple files.
            If the implementation expects a file path, it should accept it
            without file ending.
        convert_data_parallel : bool or {``'auto'``, ``'keep'``}, optional
            Whether to automatically convert the model weight names if
            :attr:`model` is a :class:`nn.DataParallel`-model but the stored
            state dict stems from a non-data-parallel model, or vice versa.

                ``'auto'`` or ``True``:
                    Auto-convert weight names, depending on the type of
                    :attr:`model`.
                ``'keep'`` or ``False``:
                    Do not convert weight names.
                    Convert to plain weight names.
        """
        path = path if path.endswith('.pt') else path + '.pt'
        self.init_model()
        map_location = ('cuda:0' if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)

        if convert_data_parallel == 'auto' or convert_data_parallel == True:
            load_state_dict_convert_data_parallel(self.model, state_dict)
        elif convert_data_parallel == 'keep' or convert_data_parallel == False:
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError("Unknown option '{}' for `convert_data_parallel`"
                             .format(convert_data_parallel))
