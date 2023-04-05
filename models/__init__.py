# Disclaimer: each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.

from models.mnist import classifier_mnist, robustifier_mnist
from models.cifar10 import classifier_cifar10, robustifier_cifar10
from models.fonts import classifier_fonts
#from models.tinyimagenet import classifier_tinyimagenet, robustifier_tinyimagenet
from models.identity import identity

Models = {'classifier_mnist': classifier_mnist,
          'robustifier_mnist': robustifier_mnist,
          'classifier_cifar10': classifier_cifar10,
          'robustifier_cifar10': robustifier_cifar10,
          #'classifier_tinyimagenet': classifier_tinyimagenet,
          #'robustifier_tinyimagenet': robustifier_tinyimagenet,
          'classifier_fonts': classifier_fonts,
          'robustifier_identity': identity}
