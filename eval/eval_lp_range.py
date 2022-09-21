import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from autoattack.autopgd_base import APGDAttack
import sys
sys.path.insert(0, '..')

from train.wide_resnet import wrn_28_10
from train.resnet_cifar import resnet18
from train.vgg import vgg16_bn
import train.activations as activations
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='wrn_28_10',
                        choices=['wrn_28_10', 'resnet18', 'vgg'])
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--preprocess', type=str, default='01',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)

    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--version', type=str, default='standard')

    args = parser.parse_args()
    num_classes = int(args.data[5:])
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')
    act = activations.__dict__[args.act]()
    # model = ResNet18()
    if args.arch == 'wrn_28_10':
        net = wrn_28_10(num_classes=num_classes, activation=act)
    elif args.arch == 'resnet18':
        net = resnet18(num_classes=num_classes, activation=act)
    elif args.arch == 'vgg16':
        net = vgg16_bn(num_classes=num_classes, activation=act)
    else:
        raise ValueError('Please use choose correct architectures.')

    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    net.load_state_dict(ckpt)
    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    model.to(device)
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=256, shuffle=False, num_workers=0)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    linf_range = np.arange(8/255, 25/255, 1/255)
    l2_range = np.arange(0.5, 2.3, 0.1)
    l1_range = np.arange(5, 23, 1)
    eps = zip(linf_range, l2_range, l1_range)
    errors_linf = np.zeros(len(linf_range))
    errors_l2 = np.zeros(len(l2_range))
    errors_l1 = np.zeros(len(l1_range))
    accs_linf = np.zeros(len(linf_range))
    accs_l2 = np.zeros(len(l2_range))
    accs_l1 = np.zeros(len(l1_range))
    for i, (linf_eps, l2_eps, l1_eps) in enumerate(eps):
        adv_linf = APGDAttack(model, norm='Linf', eps=linf_eps)
        adv_l2 = APGDAttack(model, norm='L2', eps=l2_eps)
        adv_l1 = APGDAttack(model, norm='L1', eps=l1_eps)
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            adv = adv_linf.perturb(x, y)
            pred = model(adv)
            errors_linf[i] += criterion(pred, y)
            accs_linf[i] += (pred.argmax(1) == y).sum()

            adv = adv_l2.perturb(x, y)
            pred = model(adv)
            errors_l2[i] += criterion(pred, y)
            accs_l2[i] += (pred.argmax(1) == y).sum()

            adv = adv_l1.perturb(x, y)
            pred = model(adv)
            errors_l1[i] += criterion(pred, y)
            accs_l1[i] += (pred.argmax(1) == y).sum()
    errors_linf /= len(test_loader.dataset)
    errors_l2 /= len(test_loader.dataset)
    errors_l1 /= len(test_loader.dataset)
    accs_linf /= len(test_loader.dataset)
    accs_l2 /= len(test_loader.dataset)
    accs_l1 /= len(test_loader.dataset)

    np.save(os.path.join(args.log_path, 'linf_errs.npy'), errors_linf)
    np.save(os.path.join(args.log_path, 'l2_errs.npy'), errors_l2)
    np.save(os.path.join(args.log_path, 'l1_errs.npy'), errors_l1)
    np.save(os.path.join(args.log_path, 'linf_accs.npy'), accs_linf)
    np.save(os.path.join(args.log_path, 'l2_accs.npy'), accs_l2)
    np.save(os.path.join(args.log_path, 'l1_accs.npy'), accs_l2)
