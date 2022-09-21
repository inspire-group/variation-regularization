import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.insert(0, '..')

from train.wide_resnet import wrn_28_10
from train.resnet_cifar import resnet18
from train.vgg import vgg16_bn
from train.ResNet import ResNet18
import train.activations as activations
from perceptual_advex.attacks import StAdvAttack
from perceptual_advex.attacks import JPEGLinfAttack
from perceptual_advex.attacks import ReColorAdvAttack
from perceptual_advex.attacks import FogAttack
from perceptual_advex.attacks import UARAttack
from perceptual_advex.perceptual_attacks import PerceptualPGDAttack, LagrangePerceptualAttack

from autoattack import AutoAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class JPEGL1Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=1024, **kwargs):
        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='jpeg_l1',
            bound=bound,
            **kwargs,
        )

        
class ElasticAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=0.25, **kwargs):
        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='elastic',
            bound=bound,
            **kwargs,
        )

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
                        choices=['wrn_28_10', 'resnet18', 'vgg16', 'ResNet18'])
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNette'],
                        help='Which dataset the eval is on')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--preprocess', type=str, default='meanstd',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--union_source', action='store_true')

    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--run_full', action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        elif args.data == 'ImageNette':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
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
    if args.data in ['ImageNette', 'CIFAR10']:
        num_classes = 10
    else:
        num_classes=100
    if args.data == 'ImageNette':
        net = ResNet18(num_classes=num_classes, activation=act)
    elif args.arch == 'wrn_28_10':
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
    model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    if args.data == 'ImageNette':
        size=224
        transform_chain = transforms.Compose([
        transforms.Resize(int(1.14*size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
        item = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform_chain)
    else:
        item = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=2)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.norm == 'Linf':
        source_attack = AutoAttack(model, norm=args.norm, eps=8/255)
    else:
        source_attack = AutoAttack(model, norm='L2', eps=0.5)
    attack_linf = AutoAttack(model, norm='Linf', eps=12/255)
    attack_l2 = AutoAttack(model, norm='L2', eps=1)
    attack_stadv = StAdvAttack(model, num_iterations=100)
    attack_recolor = ReColorAdvAttack(model, num_iterations=100)
    attack_jpeg = JPEGLinfAttack(model, 'cifar', bound=0.125)
    attack_jpeg1 = JPEGL1Attack(model, 'cifar', bound=64)
    attack_elastic = ElasticAttack(model, 'cifar', bound=0.25)
    attack_ppgd = PerceptualPGDAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')
    attack_lpa = LagrangePerceptualAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')

    clean_acc = 0
    source_acc = 0
    union_acc_1 = 0
    union_acc = 0
    linf_acc = 0
    l2_acc = 0
    stadv_acc = 0
    recolor_acc = 0
    jpeg_acc = 0
    jpeg_l1_acc = 0
    elastic_acc = 0
    ppgd_acc = 0
    lpa_acc = 0

    def apply_source_union(union_source, correct, source_correct):
        if union_source:
            return (correct * source_correct).float().sum()
        else:
            return correct.float().sum()

  #  lpa_acc = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        union_correct = 1
        pred = model(x).argmax(1)
        clean_acc += (y == pred).float().sum()

        adv = source_attack.run_standard_evaluation(x, y)
        pred = model(adv).argmax(1)
        source_correct = (y == pred).float()
        source_acc += source_correct.sum()
        if args.union_source:
            union_correct *= source_correct

        adv = attack_linf.run_standard_evaluation(x, y)
        pred = model(adv).argmax(1)
        linf_acc += apply_source_union(args.union_source, y == pred, source_correct)
        union_correct *= (y == pred).float()

        adv = attack_l2.run_standard_evaluation(x, y)
        pred = model(adv).argmax(1)
        l2_acc += apply_source_union(args.union_source, y == pred, source_correct)
        union_correct *= (y == pred).float()

        adv = attack_stadv(x, y)
        pred = model(adv).argmax(1)
        stadv_acc += apply_source_union(args.union_source, y == pred, source_correct)
        union_correct *= (y == pred).float()

        adv = attack_recolor(x, y)
        pred = model(adv).argmax(1)
        recolor_acc += apply_source_union(args.union_source, y == pred, source_correct)
        union_correct *= (y == pred).float()
        union_acc_1 += union_correct.sum()
        if args.run_full:
            adv = attack_jpeg(x, y)
            pred = model(adv).argmax(1)
            jpeg_acc += apply_source_union(args.union_source, y == pred, source_correct)
            union_correct *= (y == pred).float()

            adv = attack_jpeg1(x, y)
            pred = model(adv).argmax(1)
            jpeg_l1_acc += apply_source_union(args.union_source, y == pred, source_correct)
            union_correct *= (y == pred).float()

            adv = attack_elastic(x, y)
            pred = model(adv).argmax(1)
            elastic_acc += apply_source_union(args.union_source, y == pred, source_correct)
            union_correct *= (y == pred).float()

            adv = attack_ppgd(x, y)
            pred = model(adv).argmax(1)
            ppgd_acc += apply_source_union(args.union_source, y == pred, source_correct)
            union_correct *= (y == pred).float()

            adv = attack_lpa(x, y)
            pred = model(adv).argmax(1)
            lpa_acc += apply_source_union(args.union_source, y == pred, source_correct)
            union_correct *= (y == pred).float()

            union_acc += union_correct.sum()

    linf_acc /= len(test_loader.dataset)
    l2_acc /= len(test_loader.dataset)
    stadv_acc /= len(test_loader.dataset)
    recolor_acc /= len(test_loader.dataset)
    jpeg_acc /= len(test_loader.dataset)
    jpeg_l1_acc /= len(test_loader.dataset)
    elastic_acc /= len(test_loader.dataset)
    lpa_acc /= len(test_loader.dataset)
    ppgd_acc /= len(test_loader.dataset)
    union_acc /= len(test_loader.dataset)
    clean_acc /= len(test_loader.dataset)
    source_acc /= len(test_loader.dataset)
    union_acc_1 /= len(test_loader.dataset)

    with open(os.path.join(args.log_path, 'unforeseen_evals.log'), 'w') as f:
        f.write('Clean acc: {:.2%}\n'.format(clean_acc))
        f.write('Source acc: {:.2%}\n'.format(source_acc))
        f.write('Linf acc: {:.2%}\n'.format(linf_acc))
        f.write('L2 acc: {:.2%}\n'.format(l2_acc))
        f.write('StAdv acc: {:.2%}\n'.format(stadv_acc))
        f.write('Recolor acc: {:.2%}\n'.format(recolor_acc))
        f.write('Union acc 1: {:.2%}\n'.format(union_acc_1))
        f.write('---------- \n')
        f.write('Linf JPEG acc: {:.2%}\n'.format(jpeg_acc))
        f.write('L1 JPEG acc: {:.2%}\n'.format(jpeg_l1_acc))
        f.write('Elastic acc: {:.2%}\n'.format(elastic_acc))
        f.write('PPGD acc: {:.2%}\n'.format(ppgd_acc))
        f.write('LPA acc: {:.2%}\n'.format(lpa_acc))
        f.write('Union acc overall: {:.2%}\n'.format(union_acc))

