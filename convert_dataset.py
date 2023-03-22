import argparse, os, torch, multiprocessing, cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import webdataset as wds
from torch.utils.data import Dataset
import numpy as np


# base_pattern: base folder for storing the dataset
# dataset: dataset (pytorch format)
# m: average (later used for normalization)
# s: standard deviation (later used for normalization)
# n: number of classes
def create_webdataset(base_pattern, dataset, m, s, n, names):

  # mean and standard deviation
  torch.save(m, os.path.join(base_pattern, "m.pt"))
  torch.save(s, os.path.join(base_pattern, "s.pt"))
  torch.save(n, os.path.join(base_pattern, "n.pt"))
  torch.save(names, os.path.join(base_pattern, "names.pt"))

  # webdataset
  with torch.no_grad():
    pattern = os.path.join(base_pattern, f"data-%06d.tar")
    #with wds.ShardWriter(pattern, maxcount=10000) as sink:
    sink = wds.ShardWriter(pattern, maxcount=10000)
    for i, (data, label) in enumerate(dataset):
        key = "%.6d" % i
        sample = {"__key__": key,
                "ppm": data,
                "cls": label,
                "pyd": i}
        sink.write(sample)
        print("converting: %.2f%%." % (100.0 * i / len(dataset)))


        if i == 0:
          torch.save(np.shape(data), os.path.join(base_pattern, "size.pt"))

    sink.close()

  # also save the number of samples
  torch.save(i+1, os.path.join(base_pattern, "num_samples.pt"))

  return


class FontDataset(Dataset):
  def __init__(self, folder='./fonts'): #, enlargement_factor=20, binarize=False):
    self.num_fonts = 1
    self.xs = np.zeros((62 * self.num_fonts, 1, 128, 128), dtype=np.uint8) #torch.zeros((62 * self.num_fonts, 1, 128, 128)).float()
    self.ys = np.zeros(62 * self.num_fonts, dtype=np.int32) #torch.zeros(62 * self.num_fonts).long()
    fonts_list = ['comicsansms', 'couriernew', 'verdana', 'arial', 'latinmodernmonolight', 'chilanka', 'freemono', 'impact', 'jamrul', 'uroob']
    idx = 0
    for n in range(62):
      for font in fonts_list[0:self.num_fonts]:
        filename = folder + '/%.3d_' % n + font + '.png'
        img = 255 - cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY) #255 - cv2.imread(filename) # np.transpose(255 - cv2.imread(filename), (2, 0, 1)) # 255 - cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        self.xs[idx] = img
        self.ys[idx] = n# idx
        idx += 1

  def __len__(self):
    return 62 * self.num_fonts

  def __getitem__(self, idx):
    return (self.xs[idx, 0],  self.ys[idx])



def main():
  parser = argparse.ArgumentParser()

  # train
  parser.add_argument('--dataset-name',  type=str, choices=['mnist', 'fashionmnist', 'cifar10', 'tinyimagenet', 'cifar100', 'fonts'], default='mnist', help='dataset')
  parser.add_argument('--output-folder', type=str, default=None, help=' output folder')
  args = parser.parse_args()

  # Check for output folder
  if args.output_folder is None:
    print("Passing an output folder is mandatory. Returning.")
    return

  # Create output folder
  if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)
    train_folder = os.path.join(args.output_folder, 'train')
    validation_folder = os.path.join(args.output_folder, 'validation')
    test_folder = os.path.join(args.output_folder, 'test')
    os.mkdir(train_folder)
    os.mkdir(validation_folder)
    os.mkdir(test_folder)
  else:
    print("The output folder already exist. Please delete it. Returning.")
    return

  # Open the standard dataset - if you want to manage more datasets, please add your code here
  if args.dataset_name == 'mnist':
    data = datasets.MNIST("./data", train=True, download=True)
    (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95*len(data)), int(0.05*len(data))], generator=torch.Generator().manual_seed(42))
    dtest_data = datasets.MNIST("./data", train=False, download=True)
    m = torch.tensor([0.0])
    s = torch.tensor([1.0])
    n = torch.tensor([10])
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  elif args.dataset_name == 'fashionmnist':
    data = datasets.FashionMNIST("./data", train=True, download=True)
    (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95 * len(data)), int(0.05 * len(data))], generator=torch.Generator().manual_seed(42))
    dtest_data = datasets.FashionMNIST("./data", train=False, download=True)
    m = torch.tensor([0.5])
    s = torch.tensor([2.0])
    n = torch.tensor([10])
    names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  elif args.dataset_name == 'cifar10':
    data = datasets.CIFAR10("./data", train=True, download=True)
    (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95 * len(data)), int(0.05 * len(data))], generator=torch.Generator().manual_seed(42))
    dtest_data = datasets.CIFAR10("./data", train=False, download=True)
    m = torch.tensor([0.4914, 0.4822, 0.4465])
    s = torch.tensor([0.2023, 0.1994, 0.2010])
    n = torch.tensor([10])
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  elif args.dataset_name == 'tinyimagenet':
    # TODO add tinyimagenet dataset
    #data = datasets.("./data", train=True, download=True)
    #(dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95 * len(data)), int(0.05 * len(data))], generator=torch.Generator().manual_seed(42))
    #dtest_data = datasets.CIFAR10("./data", train=False, download=True)
    #m = torch.tensor([0.4914, 0.4822, 0.4465])
    #s = torch.tensor([0.2023, 0.1994, 0.2010])
    #n = torch.tensor([10])
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  elif args.dataset_name == 'fonts':
    dtrain_data = FontDataset("./fonts")
    dvalidation_data = FontDataset("./fonts")
    dtest_data = FontDataset("./fonts")
    m = torch.tensor([0.5])
    s = torch.tensor([1.0])
    n = torch.tensor([62]) # * dtrain_data.num_fonts])
    names = []
    for i in range(65, 91): names.append(chr(i))
    for i in range(97, 123): names.append(chr(i))
    for i in range(48, 58): names.append(chr(i))
  else:
    print("Incorrect dataset name. Returning.")
    return
  print("Length of the training dataset: %d." % (len(dtrain_data)))
  print("Length of the validation dataset: %d." % (len(dvalidation_data)))
  print("Length of the testing dataset: %d." % (len(dtest_data)))

  # Create the webdataset
  create_webdataset(base_pattern=train_folder, dataset=dtrain_data, m=m, s=s, n=n, names=names)
  create_webdataset(base_pattern=validation_folder, dataset=dvalidation_data, m=m, s=s, n=n, names=names)
  create_webdataset(base_pattern=test_folder, dataset=dtest_data, m=m, s=s, n=n, names=names)

  return

if __name__ == "__main__":
  main()
