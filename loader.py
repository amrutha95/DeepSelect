
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import sampler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
        
def CIFAR10_loader(batch_size=16):
  #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
  #                                   std=[0.229, 0.224, 0.225])

  train_loader = DataLoader(datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor()
                         ])), batch_size, shuffle=False)

  test_loader = DataLoader(datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                         transforms.ToTensor()
                         ])), batch_size, shuffle=False)

  loaders = {'train_loader':train_loader, 'test_loader': test_loader}
  return loaders
