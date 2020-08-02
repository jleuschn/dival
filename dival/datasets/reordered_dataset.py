from dival.datasets import Dataset


class ReorderedDataset(Dataset):
    """
    Dataset that reorders the samples of another dataset by specified index
    arrays for each part.
    """

    def __init__(self, dataset, idx):
        """
        Parameters
        ----------
        dataset : `Dataset`
            Dataset to take the samples from. Must support random access.
        idx : dict of array-like
            Indices into the original dataset for each part.
            Each array-like must have (at least) the same length as the part.
        """
        assert dataset.supports_random_access()
        self.dataset = dataset
        self.idx = idx
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.random_access = True
        self.num_elements_per_sample = (
            self.dataset.get_num_elements_per_sample())
        super().__init__(space=self.dataset.space)
        self.shape = self.dataset.get_shape()

    # use default implementation of generator

    def get_sample(self, index, part='train', out=None):
        sample = self.dataset.get_sample(
            self.idx[part][index], part=part, out=out)
        return sample

    # use default implementation of get_samples, which calls get_sample
    # (seems close to optimal as the indices do not follow a simple pattern)
