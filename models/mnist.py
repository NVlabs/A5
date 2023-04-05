# Disclaimer: each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.robustifier import Robustifier

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(25088, 512), #32768, 512)
                                   nn.ReLU(),
                                   nn.Linear(512, 10))


    def forward(self, x):

        logits = self.model(x)
        return logits


def classifier_mnist():
  return Net()


def robustifier_mnist(x_min, x_max, x_avg, x_std, x_epsilon_defense):
  convolutional_dnn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, 5, 1, 2)
  )

  return Robustifier(x_min, x_max, x_avg, x_std, x_epsilon_defense, convolutional_dnn).cuda()



