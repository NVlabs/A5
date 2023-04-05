# Disclaimer: each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x.clone()


def identity():
  return Net()



