import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class Cifar10Dataloader(object):

    def __init__(self, traindataset, testdataset, batch_size=64):
        self.traindataset = traindataset
        self.testdataset = testdataset

        # number of subprocesses to use for data loading
        self.num_workers = 0
        # how many samples per batch to load
        self.batch_size = 64
        # percentage of training set to use as validation
        valid_size = 0.2

        seed = 1
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        if cuda:
            self.batch_size = batch_size
            self.num_workers = 4
            self.pin_memory = True
        else:
            self.shuffle = True
            self.batch_size = batch_size

        print(self.batch_size)

        # obtain training indices that will be used for validation
        num_train = len(self.traindataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)  # For reproducibility

    def gettraindataloader(self):
        return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True)

    def gettestdataloader(self):
        return torch.utils.data.DataLoader(dataset=self.testdataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True)
