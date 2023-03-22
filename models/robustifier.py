import torch

def zx2x_rob(z, x, x_epsilon, xD_min, xD_max):
  dx = x_epsilon.view(1, -1, 1, 1) * (-1.0 + 2.0 * torch.sigmoid(z))
  x_rob = torch.clamp(x + dx, min=xD_min.view(1, -1, 1, 1), max=xD_max.view(1, -1, 1, 1))

  return x_rob


# Be careful - inversion is far from perfect because of clipping and quantization
def xx_rob2z(x, x_rob, x_epsilon):
  dx = x_rob - x
  z = torch.log(torch.clamp(x_epsilon + dx, min=1e-12) / torch.clamp(x_epsilon - dx, min=1e-12))

  return z


class Robustifier(torch.nn.Module):
  def __init__(self, x_min, x_max, x_avg, x_std, x_epsilon_defense, convolutional_dnn=torch.nn.Sequential()):
    super(Robustifier, self).__init__()
    self.x_min = x_min
    self.x_max = x_max
    self.normalized_x_min = (self.x_min - x_avg) / x_std
    self.normalized_x_max = (self.x_max - x_avg) / x_std
    self.x_epsilon_defense = x_epsilon_defense
    self.normalized_x_epsilon_defense = x_epsilon_defense / x_std
    self.convolutional_dnn = convolutional_dnn

  # accept in intput x (between 0.0 and 1.0)
  def forward(self, normalized_x):
    z = self.convolutional_dnn(normalized_x) # z can be used to generate the
    x_rob = zx2x_rob(z, normalized_x, self.normalized_x_epsilon_defense, self.normalized_x_min, self.normalized_x_max)

    return x_rob
