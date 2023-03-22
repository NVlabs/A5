import torch
import torch.nn as nn
import torch.nn.functional as F
from models.robustifier import Robustifier

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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
                                   nn.Linear(32768, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 10))


    def forward(self, x):

        logits = self.model(x)
        return logits


def classifier_cifar10():
  return Net()


def robustifier_cifar10(x_min, x_max, x_avg, x_std, x_epsilon_defense):
  convolutional_dnn = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 5, 1, 2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 3, 5, 1, 2))

  return Robustifier(x_min, x_max, x_avg, x_std, x_epsilon_defense, convolutional_dnn).cuda()



