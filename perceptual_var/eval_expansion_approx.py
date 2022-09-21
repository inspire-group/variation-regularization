import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from perceptual_advex.var_utils import StAdvVariation, ReColorAdvVariation
#from robustness.cifar_models import ResNet18 as resnet18

import sys
sys.path.insert(0, '..')

from train.wide_resnet import wrn_28_10
from train.resnet_cifar import resnet18
from train.vgg import vgg16_bn
import train.activations as activations

from train.utils import *
from train.utils_var import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--data-dir', default='../../robust_activation/data_dir/cifar10', type=str)
    parser.add_argument('--models-dir', default='resnet_models')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epsilon-source', default=8, type=float)
    parser.add_argument('--epsilon-target', default=16, type=float)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--use-every', default=1, type=int)
    parser.add_argument('--randomize', default=0, type=int)
    parser.add_argument('--logits-var', action='store_true')
    return parser.parse_args()

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu)/std

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.cuda().float(), 'target': y.cuda().long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias)

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict.keys():
        state_dict = state_dict['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            k = k[7:]
        if 'fc' in k:
            k = k.replace('fc', 'classifier')
        
        new_state_dict[k] = v
    return new_state_dict

def main():
    args = get_args()

    transforms = [Crop(32, 32), FlipLR()]
    dataset = cifar10(args.data_dir)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    epsilon_source = (args.epsilon_source / 255.)
    epsilon_target = (args.epsilon_target / 255.)
    
    act = activations.__dict__[args.activation]()
    base_dir = args.models_dir
    
    max_cond_num = 0
    for folder in os.listdir(base_dir):
        if 'l2' not in folder:
            continue
        path = os.path.join(base_dir, folder)
        for f in os.listdir(path):
            if 'model' in f or 'ckpt' in f:
                if 'ckpt' in f:
                    epoch = int(f.split('.')[0])
                    if not (epoch + 1) % args.use_every == 0:
                        continue
                f = os.path.join(path, f)
                print('Evaluating {}'.format(f))
                model = resnet18(num_classes=10, activation=nn.ReLU())
                model.load_state_dict(filter_state_dict(torch.load(f)))
                model = nn.DataParallel(model).cuda()
                model.eval()
                
                model_cond_num = 0

                for i, batch in enumerate(test_batches):
                    X, y = batch['input'], batch['target']
                    # take the local approx around X --> just need the get the gradient
                    X.requires_grad = True
                    grads = []
                    for i in range(10):
                        pred = model(X)
                        selector = torch.zeros_like(pred).cuda()
                        selector[:, i] = 1.0
                        score = pred
                        pred.backward(selector)
                        grad = X.grad.detach().reshape(X.size(0), 1, -1)
                        grads.append(grad)
                        X.grad=None
                    # compute condition number
                    grads = torch.cat(grads, dim=1)
                    S = torch.linalg.svdvals(grads)
                    #print(S.size(0), X.size(0))
                    max_eig = S.max(dim=-1)[0]
                    min_eig = S.min(dim=-1)[0]
                    check = max_eig[(min_eig == 0).nonzero(as_tuple=True)]
                    assert(len(check) == 0 or check.max().item() == 0)
                    model_cond_num += (max_eig[min_eig.nonzero(as_tuple=True)] / min_eig[min_eig.nonzero(as_tuple=True)]).sum().item() / len(test_batches.dataloader.dataset)
                print('model_cond_num', model_cond_num)
                
                max_cond_num = max(max_cond_num, model_cond_num)
    print('max_cond_num', max_cond_num)
    print('estimated expansion slope', max_cond_num * args.epsilon_target / args.epsilon_source)


if __name__ == '__main__':
    main() 
