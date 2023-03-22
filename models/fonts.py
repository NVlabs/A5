import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # 128 -> 64
          nn.ReLU(),
          nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # 64 -> 32
          nn.ReLU(),
          nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 32 -> 16
          nn.ReLU(),
          nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 16 -> 8
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 62))

    def forward(self, x):
        logits = self.model(x)

        return logits

def classifier_fonts():
  return Net()