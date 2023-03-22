import argparse, sys, time
import pdb

import torch.nn
from utils.utils import *
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
from auto_LiRPA.utils import MultiAverageMeter
from tensorboardX import SummaryWriter
from models.robustifier import zx2x_rob, xx_rob2z
from autoattack import AutoAttack

def parse_argumments():
  # parse argument
  parser = argparse.ArgumentParser()

  # tasks
  parser.add_argument('--train-prototypes', action='store_true', help='train prototypes')
  parser.add_argument('--train-robustifier', action='store_true', help='train robustifier')
  parser.add_argument('--train-classifier', action='store_true', help='train classifier')
  parser.add_argument('--test', action='store_true', help='test')
  parser.add_argument('--no-autoattack', action='store_true', help='do not use autoattack for testing (faster)')

  # archs
  parser.add_argument('--robustifier-arch', type=str, choices=['mnist', 'cifar10', 'tinyimagenet', 'identity'], default='mnist', help='robustifier architecture')
  parser.add_argument('--acquisition-arch', type=str, choices=['identity', 'camera'], default='identity', help='acquisition device architecture')
  parser.add_argument('--classifier-arch', type=str, choices=['mnist', 'cifar10', 'tinyimagenet', 'fonts'], default='mnist', help='classifier architecture')

  # dataset
  parser.add_argument('--training-dataset-folder', type=str, default=None, help='training dataset folder (default: None)')
  parser.add_argument('--validation-dataset-folder', type=str, default=None, help='validation dataset folder (default: None)')
  parser.add_argument('--test-dataset-folder', type=str, default=None, help='testing dataset folder (default: None)')
  parser.add_argument('--prototypes-dataset-folder', type=str, default=None, help='dataset with trained prototypes (default: None)')

  # training params
  parser.add_argument('--batch-size', type=int, default=100, help='batch size (default 100)')
  parser.add_argument('--epochs', type=int, default=100, help='training epochs (default 100)')
  parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default 1e-3)')
  parser.add_argument("--lr-scheduler-milestones", type=int, nargs='+', default=[25, 42], help='list of epoch milestones to decrease lr (default [25, 42])')
  parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='gamma for lr scheduler (default 0.1)')
  parser.add_argument("--x-epsilon-attack-scheduler-name", type=str, default="SmoothedScheduler", choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon attack x scheduler (default SmoothScheduler)')
  parser.add_argument("--x-epsilon-attack-scheduler-opts", type=str, default='start=3,length=18,mid=0.3', help='options for epsilon attack x scheduler (default ''start=3,length=18,mid=0.3''')
  parser.add_argument("--x-augmentation-mnist", action='store_true', help='augment x during training, for mnist')
  parser.add_argument("--x-augmentation-cifar10", action='store_true', help='augment x during training, for cifar')
  parser.add_argument("--save-interval", type=int, default=5, help="interval for saving model (in epochs)")

  parser.add_argument("--batch-multiplier", type=int, default=1, help='batch multiplicative factor (reduces memory consumption) - default: 1')
  parser.add_argument("--test-multiplier", type=int, default=1, help='test multiplicative factore (reduces the variance of test) - default: 1' )

  # load
  parser.add_argument('--load-classifier', type=str, default=None, help='if provided, load the classifier indicated here')
  parser.add_argument('--load-robustifier', type=str, default=None, help='if provided, load the robustifier indicated here')

  # log and save
  parser.add_argument('--log-dir', type=str, default='log/', help='log folder')
  #parser.add_argument('--save-w-ratio', type=float, default=1.0, help='ratio of the modified dataset to be saved in the log dir (can be used as dataset in future - also save debug images)')

  # attack params
  parser.add_argument('--x-epsilon-attack-training', type=float, default=0.1, help='epsilon for MitM attack during training')
  parser.add_argument('--x-epsilon-attack-testing', type=float, default=0.1, help='epsilon for MitM attack during testing')
  parser.add_argument('--w-epsilon-attack-training', type=float, default=0.1, help='epsilon for physical attack during training')
  parser.add_argument('--w-epsilon-attack-testing', type=float, default=0.1, help='epsilon for physical attack during testing')

  # robustifier
  parser.add_argument('--x-epsilon-defense', type=float, default=0.1, help='epsilon for defense (x)')
  parser.add_argument('--w-epsilon-defense', type=float, default=0.5, help='epsilon for defense (w)')

  # auto_LiRPA parameters
  parser.add_argument("--bound-type", type=str, default="CROWN-IBP", choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis')

  # verbose
  parser.add_argument('--verbose', action='store_true', help='verbose')

  args = parser.parse_args()

  return args


def compute_predictions_and_loss(classifier, normalized_x, normalized_x_min, normalized_x_max, normalized_x_epsilon, f, bound_type, y, num_classes, ce):

  # quick compute for easy case
  if f == 1:
    prediction = classifier(normalized_x)  # natural prediction
    reg_ce = ce(prediction, y)
    reg_err = torch.sum(torch.argmax(prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)
    ver_ce = reg_ce
    ver_err = reg_err
    loss = reg_ce
    return (prediction, reg_ce, reg_err, ver_ce, ver_err, loss)

  # prediction: lower and upper bounds (auto_LiRPA) - also use the linear comb in the last layer
  ptb = PerturbationLpNorm(norm=np.inf, eps=normalized_x_epsilon, x_L=torch.max(normalized_x - normalized_x_epsilon.view(1, -1, 1, 1), normalized_x_min.view(1, -1, 1, 1)), x_U=torch.min(normalized_x + normalized_x_epsilon.view(1, -1, 1, 1), normalized_x_max.view(1, -1, 1, 1)))
  data = BoundedTensor(normalized_x, ptb)

  c = torch.eye(num_classes).type_as(data)[y].unsqueeze(1) - torch.eye(num_classes).type_as(data).unsqueeze(0).cuda()
  I = (~(y.data.unsqueeze(1) == torch.arange(num_classes).type_as(y.data).unsqueeze(0))).cuda()
  c = (c[I].view(data.size(0), num_classes - 1, num_classes)).cuda()

  # prediction: clean data
  prediction = classifier(data)  # natural prediction
  reg_ce = ce(prediction, y)
  reg_err = torch.sum(torch.argmax(prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)

  if bound_type == 'CROWN-IBP':
    ilb, iub = classifier.compute_bounds(IBP=True, C=c, method=None)
    if f < 1e-5:
      lb = ilb
    else:
      clb, cub = classifier.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
      lb = clb * f + ilb * (1 - f)
  else:
    lb, ub = classifier.compute_bounds(x=(data,), method=bound_type, C=c)

  lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
  fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
  ver_ce = ce(-lb_padded, fake_labels)
  ver_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)

  # loss
  loss = reg_ce * f + ver_ce * (1.0 - f)

  return (prediction, reg_ce, reg_err, ver_ce, ver_err, loss)


def vprint(s, verbose):
  if verbose:
    print(s)


def main():

  ###################################################################################################
  # Parser

  args = parse_argumments()
  logger = LogDir(args.log_dir)
  logger.parser(sys, args)

  ###################################################################################################
  # Prepare dataset

  # TODO load only the usefull datasets
  (train_loader, train_avg, train_std, train_num, train_siz, train_num_samples, train_width, train_height, train_num_channels, train_names) = open_dataset(args.training_dataset_folder, args.batch_size)
  (validation_loader, validation_avg, validation_std, validation_num, validation_siz, validation_validation_num_samples, validation_width, validation_height, validation_num_channels, validation_names) = open_dataset(args.validation_dataset_folder, args.batch_size, shuffle=False)
  (test_loader, test_avg, test_std, test_num, test_siz, test_num_samples, test_width, test_height, test_num_channels, test_names) = open_dataset(args.test_dataset_folder, args.batch_size, shuffle=False)
  (prototypes_loader, prototypes_avg, prototypes_std, prototypes_num, prototypes_siz, prototypes_num_samples, prototypes_width, prototypes_height, prototypes_num_channels, prototypes_names) = open_dataset(args.prototypes_dataset_folder, args.batch_size, shuffle=False)
  if not(train_loader is None):
    avg, std, num, width, height, num_channels, names = train_avg, train_std, train_num, train_width, train_height, train_num_channels, train_names
  elif not(validation_loader is None):
    avg, std, num, width, height, num_channels, names = validation_avg, validation_std, validation_num, validation_width, validation_height, validation_num_channels, validation_names
  elif not(test_loader is None):
    avg, std, num,  width, height, num_channels, names = test_avg, test_std, test_num, test_width, test_height, test_num_channels, test_names
  normalized_x_min = (0.0 - avg) / std
  normalized_x_max = (1.0 - avg) / std
  vprint("Created datasets and loader. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)

  # normalize
  normalize = transforms.Normalize(avg, std)

  ###################################################################################################
  # Prepare and load models

  # models
  if args.acquisition_arch == 'identity':
    tr1 = None
    tr2 = None
    def acquisition(x, tr1, tr2):
      return x
  elif args.acquisition_arch == 'camera':
    tr1 = transforms.Compose([transforms.RandomCrop(128, 5, fill=0.0),
                             transforms.RandomRotation(5, fill=0.0),
                             transforms.RandomPerspective(distortion_scale=0.25, p=0.99, interpolation=2, fill=0.0)])
    tr2 = transforms.Compose([transforms.GaussianBlur(kernel_size=9, sigma=(0.01, 1.0)),
                             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])

    def acquisition(x, tr1, tr2):
      y = torch.clamp(tr2((1.0 - tr1(x) + torch.randn(x.size()).cuda() * (0.001 + 0.099 * torch.rand(1).cuda())).expand(-1, 3, -1, -1)), min=0.0, max=1.0)
      return y

  (robustifier_ori, classifier_ori) = create_models(robustifier_arch=args.robustifier_arch, x_min=0.0, x_max=1.0, x_avg=avg, x_std=std, x_epsilon_defense=args.x_epsilon_defense, robustifier_filename=args.load_robustifier, classifier_arch=args.classifier_arch, classifier_filename=args.load_classifier)
  dummy_input = acquisition(torch.zeros((2, num_channels, height, width)).cuda(), tr1, tr2)
  robustifier = robustifier_ori.cuda() # no need to use a bounded model! # BoundedModule(robustifier_ori, dummy_input, bound_opts={'relu': 'same-slope', 'conv_mode': 'patches'}, device='cuda:0')
  classifier = BoundedModule(classifier_ori, dummy_input, bound_opts={'relu': 'same-slope', 'conv_mode': 'patches'}, device='cuda')


  if args.x_augmentation_mnist:
    augmentation = transforms.Compose([transforms.RandomCrop(28, 4), transforms.RandomRotation(10)]) # TODO this works for MNIST only
  elif args.x_augmentation_cifar10:
    augmentation = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)])  # TODO this works for CIFAR10 only
  else:
    augmentation = torch.nn.Sequential()

  vprint("Created models. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)

  # TODO when training to defende against a physical attack, we must consider the camera + robustifier + classifier as a unique classifier

  # init
  ce = torch.nn.CrossEntropyLoss()

  ###################################################################################################
  # Train

  # TODO solve the ResourceWarning: unclosed file <_io.BufferedReader name='./webdataset_mnist/train/data-000003.tar'> yield sample warning!

  if (args.train_prototypes) or (args.train_robustifier) or (args.train_classifier):

    # z
    z = torch.zeros((train_num_samples, num_channels, height, width), dtype=torch.float32, requires_grad=True, device='cuda')

    # optimizer
    parameters_list = []
    if args.train_prototypes:
      parameters_list = parameters_list + [z, ]
    if args.train_robustifier:
      parameters_list = parameters_list + list(robustifier_ori.parameters())
    if args.train_classifier:
      parameters_list = parameters_list + list(classifier_ori.parameters())
    opt = torch.optim.RMSprop(parameters_list, lr=args.lr)
    opt.zero_grad()

    # schedulers
    x_epsilon_attack_scheduler = eval(args.x_epsilon_attack_scheduler_name)(args.x_epsilon_attack_training, args.x_epsilon_attack_scheduler_opts.replace(" ", ""))
    x_epsilon_attack_scheduler.set_epoch_length(int(train_num_samples / args.batch_size))
    x_epsilon_attack_scheduler.step_batch()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_scheduler_milestones, gamma=args.lr_scheduler_gamma, verbose=False)

    # tensorboard
    writer = SummaryWriter(logger.tb_dir)

    # init
    t0 = time.time()
    freq = 0
    global_step = 0
    best_ver_err = 1.0
    x_epsilon_attack_scheduler.train()

    batch_multiplier = args.batch_multiplier
    batch_multiplier_index = 0

    # epochs
    for epoch in range(args.epochs):

      # go through the entire dataset
      if not(train_loader is None):
        for (i, (w, y, idxs)) in enumerate(train_loader):

          # to GPU
          w = w.cuda().float()
          y = y.cuda()
          idxs = idxs.cuda()
          vprint("Loaded batch data w. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

          # robustify w (proptotyes P, also used for A5/O)
          w_rob = zx2x_rob(z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), x=w, x_epsilon= torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda())
          vprint("Robustified batch data w. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

          # camera acquisition x = A(w), also include normalization
          x = augmentation(acquisition(w_rob, tr1, tr2))
          normalized_x = normalize(x)
          vprint("Augmented batch data w. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

          # robustify x (robustifier R)
          normalized_x_rob = robustifier(normalized_x)
          vprint("Robustified batch data x. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

          # predictions (classifier C) and loss computation
          x_epsilon_attack = x_epsilon_attack_scheduler.get_eps()  # get the current value of the attacking epsilon
          normalized_x_epsilon_attack = x_epsilon_attack / std  # Notice this can be a vector, not just a scalar
          f = (x_epsilon_attack_scheduler.get_max_eps() - x_epsilon_attack) / np.max((x_epsilon_attack_scheduler.get_max_eps(), 1e-12))  # this factor is used to mix the regular and worst case entropy in the loss
          (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(classifier=classifier,
                                                                                              normalized_x=normalized_x_rob,
                                                                                              normalized_x_min=normalized_x_min,
                                                                                              normalized_x_max=normalized_x_max,
                                                                                              normalized_x_epsilon=normalized_x_epsilon_attack,
                                                                                              f=f,
                                                                                              bound_type=args.bound_type,
                                                                                              y=y,
                                                                                              num_classes=num.numpy()[0],
                                                                                              ce=ce)
          vprint("Classified batch data x and computed loss. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

          # loss backward and step
          loss.backward()
          vprint("Backward done. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)
          batch_multiplier_index += 1
          if np.mod(batch_multiplier_index, batch_multiplier) == 0:
            opt.step()
            opt.zero_grad()
            batch_multiplier_index = 0
          vprint("Step done. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)

          # step info
          with torch.no_grad():
            psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
            x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
            psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1))**2.0 + 1e-12).mean()))

            # tensorboard log
            writer.add_scalars('Loss', {'reg ce [training]': reg_ce, 'ver ce [training]': ver_ce, 'loss [training]': loss}, global_step=global_step)
            writer.add_scalars('Loss - f', {'f': f}, global_step=global_step)
            writer.add_scalars('Error', {'reg [training]': reg_err, 'ver [training]': ver_err}, global_step=global_step)
            writer.add_scalars('Epsilon x', {'[training]': x_epsilon_attack, '[testing]': args.x_epsilon_attack_testing}, global_step=global_step)
            writer.add_scalars('PSNR', {'w [training]': psnr_w, 'x [training]': psnr_x}, global_step=global_step)

            if np.mod(i, 10) == 0:
              print("Epoch %.4d batch %.5d eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB." % (epoch, i, x_epsilon_attack, f, reg_ce, reg_err*100.0, ver_ce, ver_err*100.0, loss, psnr_x, psnr_w))

          # updates after each step
          x_epsilon_attack_scheduler.step_batch(verbose=True)
          global_step += 1

          # timing (log only)
          t1 = time.time()
          freq = 0.99 * freq + 0.01 * 1.0 / (t1 - t0)
          if np.mod(i, 10) == 0:
            vprint("Iteration time: %.2fms [%.2fHz - filtered: %.2fHz]" % ((t1 - t0) * 1000.0, 1.0 / (t1 - t0), freq), args.verbose)
          t0 = time.time()

      # validation
      if not (validation_loader is None):
        meter = MultiAverageMeter()
        with torch.no_grad():
          epsilon_x_attack = args.x_epsilon_attack_testing
          normalized_x_epsilon_attack = epsilon_x_attack / std
          for (i, (w, y, idxs)) in enumerate(validation_loader):
            w = w.cuda()
            y = y.cuda()
            idxs = idxs.cuda()
            w_rob = zx2x_rob(z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), x=w, x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda())
            x = acquisition(w_rob, tr1, tr2)
            normalized_x = normalize(x)
            normalized_x_rob = robustifier(normalized_x)
            (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(classifier=classifier,
                                                                                                normalized_x=normalized_x_rob,
                                                                                                normalized_x_min=normalized_x_min,
                                                                                                normalized_x_max=normalized_x_max,
                                                                                                normalized_x_epsilon=normalized_x_epsilon_attack,
                                                                                                f=f,
                                                                                                bound_type=args.bound_type,
                                                                                                y=y,
                                                                                                num_classes=num.numpy()[0],
                                                                                                ce=ce)
            meter.update('reg_ce', reg_ce, args.batch_size)
            meter.update('reg_err', reg_err, args.batch_size)
            meter.update('ver_ce', ver_ce, args.batch_size)
            meter.update('ver_err', ver_err, args.batch_size)
            meter.update('loss', loss, args.batch_size)
            psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
            x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
            psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1))**2.0 + 1e-12).mean()))
            meter.update('psnr_x', psnr_x, args.batch_size)
            meter.update('psnr_w', psnr_w, args.batch_size)

            # save images in tensorboard for debugging
            if i == 0:
              if args.train_prototypes:
                for n in range(np.min([10, args.batch_size])):
                  img = torch.cat((w[n], torch.clamp(0.5 + (w_rob[n] - w[n]) / args.w_epsilon_defense, min=0.0, max=1.0), w_rob[n]), dim=2)
                  writer.add_image('Prototye %d' % (n), img, global_step=global_step)

          s = "Epoch %.4d [Validation] eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB." % (epoch, epsilon_x_attack, f, meter.avg('reg_ce'), meter.avg('reg_err') * 100.0, meter.avg('ver_ce'), meter.avg('ver_err') * 100.0, meter.avg('loss'), meter.avg('psnr_x'), meter.avg('psnr_w'))
          print(s)

        # Write validation log
        if epoch == 0:
          ff = open(os.path.join(logger.eval_dir, 'train_eval.txt'), 'w')
        else:
          ff = open(os.path.join(logger.eval_dir, 'train_eval.txt'), 'a')
        ff.write(s + '\n')
        ff.close()

        # tensorboard log (validation and lr)
        writer.add_scalars('Loss', {'reg ce [validation]': meter.avg('reg_ce'), 'ver ce [validation]': meter.avg('ver_ce'), 'loss [validation]': meter.avg('loss')}, global_step=global_step)
        writer.add_scalars('Error', {'reg [validation]': meter.avg('reg_err'), 'ver [validation]': meter.avg('ver_err')}, global_step=global_step)
        writer.add_scalars('PSNR', {'w [validation]': meter.avg('psnr_w'), 'x [validation]': meter.avg('psnr_x')}, global_step=global_step)
        writer.add_scalars('LR', {'lr': lr_scheduler.get_last_lr()[0]}, global_step=global_step)

        # save model if best in evaluation
        if best_ver_err > meter.avg('ver_err'):
          best_ver_err = meter.avg('ver_err')
          if args.train_classifier:
            torch.save({'state_dict': classifier_ori.model.state_dict(), 'epoch': epoch}, logger.model_dir + '/classifier_best')
          if args.train_robustifier:
            torch.save({'state_dict': robustifier_ori.state_dict(), 'epoch': epoch}, logger.model_dir + '/robustifier_best')

      # save model at the end of each 5 epochs
      if np.mod(epoch + 1, args.save_interval) == 0:
        if args.train_classifier:
          torch.save({'state_dict': classifier_ori.model.state_dict(), 'epoch': epoch}, logger.model_dir + '/classifier_epoch_%.4d' % (epoch))
        if args.train_robustifier:
          torch.save({'state_dict': robustifier_ori.state_dict(), 'epoch': epoch}, logger.model_dir + '/robustifier_epoch_%.4d' % (epoch))

      # lr scheduler step
      lr_scheduler.step()

    # tensorboard
    writer.close()

  # save modified w at the end of training (if needed)
  if args.train_prototypes:

    base_pattern = logger.w_dir
    num_samples = int(train_num_samples)
    torch.save(avg, os.path.join(base_pattern, "m.pt"))
    torch.save(std, os.path.join(base_pattern, "s.pt"))
    torch.save(num, os.path.join(base_pattern, "n.pt"))
    torch.save(num_samples, os.path.join(base_pattern, "num_samples.pt"))
    torch.save(names, os.path.join(base_pattern, "names.pt"))

    # webdataset and images saving
    with torch.no_grad():
      pattern = os.path.join(base_pattern, f"data-%06d.tar")
      sink = wds.ShardWriter(pattern, maxcount=10000)
      for (i, data) in enumerate(train_loader.dataset.iterator()):
        idx = data[2]
        key = "%.6d" % idx
        w = data[0].cuda()
        w_rob = zx2x_rob(z=z[idx:idx+1], x=w, x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda()).squeeze()
        if w_rob.size(0) == 3:
          w_rob = w_rob.permute(1, 2, 0)
        sample = {"__key__": key,
                  "ppm": (w_rob * 255).cpu().numpy().astype(np.uint8),
                  "cls": data[1],
                  "pyd": idx}
        sink.write(sample)
        if np.mod(i, 100):
          print("saving: %.2f%%." % (100.0 * i / num_samples))

        if i == 0:
          torch.save(np.shape(data[0]), os.path.join(base_pattern, "size.pt"))

      sink.close()

  ###################################################################################################
  # Test

  if args.test and not(test_loader is None):

    # Initialize auto-attack - this is always testing mitm attack
    forward_pass = torch.nn.Sequential(normalize, classifier_ori.model)
    adversary = AutoAttack(forward_pass, norm='Linf', eps=args.x_epsilon_attack_testing, version='standard')

    # z
    z = torch.zeros((test_num_samples, num_channels, height, width), dtype=torch.float32, requires_grad=False, device='cuda')
    if not (prototypes_loader is None):
      # Since the w_rob may have been shuffled... let's do this.
      # First I load all the ws in z...
      for (i, (w, y, idxs)) in enumerate(test_loader):
        for n in range(w.size(0)):
          idx = idxs[n]
          z[idx] = w[n]
      # Then I load the robustified w
      w_robs = z.clone()
      for (i, (w_rob, y, idxs)) in enumerate(prototypes_loader):
        for n in range(w_rob.size(0)):
          idx = idxs[n]
          w_robs[idx] = w_rob[n]
      z = xx_rob2z(z, w_robs, args.w_epsilon_defense)
      del w_rob
    f = 0.0

    meter = MultiAverageMeter()
    with torch.no_grad():
      x_epsilon_attack = args.x_epsilon_attack_testing
      normalized_x_epsilon_attack = x_epsilon_attack / std
      for j in range(args.test_multiplier):
        for (i, (w, y, idxs)) in enumerate(test_loader):

          w = w.cuda()
          y = y.cuda()
          idxs = idxs.cuda()

          # Be sure to use the same dataset in training and testing
          w_rob = zx2x_rob(z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), x=w, x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda())
          x = acquisition(w_rob, tr1, tr2)
          normalized_x = normalize(x)
          normalized_x_rob = robustifier(normalized_x)
          (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(classifier=classifier,
                                                                                              normalized_x=normalized_x_rob,
                                                                                              normalized_x_min=normalized_x_min,
                                                                                              normalized_x_max=normalized_x_max,
                                                                                              normalized_x_epsilon=normalized_x_epsilon_attack,
                                                                                              f=f,
                                                                                              bound_type=args.bound_type,
                                                                                              y=y,
                                                                                              num_classes=num.numpy()[0],
                                                                                              ce=ce)

          meter.update('reg_ce', reg_ce, args.batch_size)
          meter.update('reg_err', reg_err, args.batch_size)
          meter.update('ver_ce', ver_ce, args.batch_size)
          meter.update('ver_err', ver_err, args.batch_size)
          meter.update('loss', loss, args.batch_size)
          psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
          x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
          psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1)) ** 2.0 + 1e-12).mean()))
          meter.update('psnr_x', psnr_x, args.batch_size)
          meter.update('psnr_w', psnr_w, args.batch_size)

          # auto-attack (find adversarial atttacks - unfortunately the error seems not to be logged, so I have to run inference again!!!)
          if args.no_autoattack:
            adv_err = np.nan
          else:
            x_adv = adversary.run_standard_evaluation(x_rob, y, bs=args.batch_size)
            prediction = classifier_ori(normalize(x_adv))
            adv_err = torch.sum(torch.argmax(prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)
          meter.update('adv_err', adv_err, args.batch_size)

          s = "[Testing] eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB [autoattack_err %.2f%%]." % (x_epsilon_attack, f, meter.avg('reg_ce'), meter.avg('reg_err') * 100.0, meter.avg('ver_ce'), meter.avg('ver_err') * 100.0, meter.avg('loss'), meter.avg('psnr_x'), meter.avg('psnr_w'), meter.avg('adv_err') * 100.0)
          print(s)

      s = "[Testing] eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB [autoattack_err %.2f%%]." % (x_epsilon_attack, f, meter.avg('reg_ce'), meter.avg('reg_err') * 100.0, meter.avg('ver_ce'), meter.avg('ver_err') * 100.0, meter.avg('loss'), meter.avg('psnr_x'), meter.avg('psnr_w'), meter.avg('adv_err') * 100.0)
      print(s)

    # Write validation log
    ff = open(os.path.join(logger.eval_dir, 'test_eval.txt'), 'w')
    ff.write(s + '\n')
    ff.close()

  ###################################################################################################
  # Return

  return 0


if __name__=="__main__":
  import warnings
  warnings.simplefilter('ignore')

  main()