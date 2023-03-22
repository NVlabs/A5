import os, torch, multiprocessing, glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import webdataset as wds
import multiprocessing

class LogDir():
  def __init__(self, log_dir):
    self.log_dir = log_dir
    self.model_dir = self.log_dir + "/models"
    self.eval_dir = self.log_dir + "/eval"
    self.tb_dir = self.log_dir + "/tb"
    self.imgs_dir = self.log_dir + "/imgs"
    self.w_dir = self.log_dir + "/w"
    if not os.path.exists(self.log_dir):
      os.mkdir(self.log_dir)
    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)
    if not os.path.exists(self.eval_dir):
      os.mkdir(self.eval_dir)
    if not os.path.exists(self.imgs_dir):
      os.mkdir(self.imgs_dir)
    if not os.path.exists(self.w_dir):
      os.mkdir(self.w_dir)

  # write
  def parser(self, sys, args):
    if args.test:
      log_file = open(self.log_dir + '/parser_test.txt', 'w')
    else:
      log_file = open(self.log_dir + '/parser.txt', 'w')
    for k in args.__dict__:
      if args.__dict__[k] is not None:
        s = k + " " + str(args.__dict__[k])
        log_file.write(s + "\n")
        print(s)
    log_file.close()


def identity(x):
  return x


def open_dataset(dataset_folder, batch_size, shuffle=True):

  # None?
  if dataset_folder is None:
    return (None, None, None, None, None, None, None, None, None, None)

  # load m, s, n
  avg = torch.load(os.path.join(dataset_folder, 'm.pt')).cuda()
  std = torch.load(os.path.join(dataset_folder, 's.pt')).cuda()
  num = torch.load(os.path.join(dataset_folder, 'n.pt')) # num classes
  siz = torch.load(os.path.join(dataset_folder, 'size.pt')) # size of the images
  num_samples = torch.load(os.path.join(dataset_folder, 'num_samples.pt'))
  names = torch.load(os.path.join(dataset_folder, 'names.pt'))
  chn = len(avg)
  if chn == 1:
    decode_pattern = 'torchl8'
    def preprocessing(x):
      x = (x.float() / 255.0).unsqueeze(0)
      return x
  elif chn == 3:
    decode_pattern = 'torchrgb8'
    def preprocessing(x):
      x = (x.float() / 255.0)
      return x
  else:
    print("Bad format in the average used for the dataset. Returning.")
    return

  # how many .tar files
  num_tar_files = len(glob.glob(os.path.join(dataset_folder, '*.tar')))
  url = "%s/data-{000000..%d}.tar" % (dataset_folder, num_tar_files-1)
  if shuffle == True:
    dataset = wds.WebDataset(url, shardshuffle=True).shuffle(1000, initial=700).decode(decode_pattern).to_tuple('ppm', 'cls', 'pyd').map_tuple(preprocessing, identity, identity)
  else:
    dataset = wds.WebDataset(url, shardshuffle=True).decode(decode_pattern).to_tuple('ppm', 'cls', 'pyd').map_tuple(preprocessing, identity, identity)

  # create loader
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count(), drop_last=True, pin_memory=True)

  avg = avg.cuda()
  std = std.cuda()
  num_channels = len(avg)

  height = siz[0]
  width = siz[1]

  # return
  return (loader, avg, std, num, siz, num_samples, width, height, num_channels, names)


def create_models(robustifier_arch, x_min, x_max, x_avg, x_std, x_epsilon_defense, robustifier_filename, classifier_arch, classifier_filename):

  # Create archs...
  classifier_ori = models.Models['classifier_' + classifier_arch]()
  if robustifier_arch == 'identity':
    robustifier_ori = models.Models['robustifier_' + robustifier_arch]() #robustifier_identity()
  else:
    robustifier_ori = models.Models['robustifier_' + robustifier_arch](x_min=x_min, x_max=x_max, x_avg=x_avg, x_std=x_std, x_epsilon_defense=x_epsilon_defense)

  # Load weights
  if not (robustifier_filename is None):
    checkpoint = torch.load(robustifier_filename, map_location=torch.device('cpu'))
    robustifier_ori.load_state_dict(checkpoint['state_dict'])
  if not (classifier_filename is None):
    checkpoint = torch.load(classifier_filename, map_location=torch.device('cpu'))
    classifier_ori.model.load_state_dict(checkpoint['state_dict'])

  return (robustifier_ori, classifier_ori)


#def create_augmenter(camera_arch, augmenter_arch):
#  return camera_augmenter, x_augmenter
