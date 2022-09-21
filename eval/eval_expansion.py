import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

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
    parser.add_argument('--epsilon-source', default=8, type=int)
    parser.add_argument('--epsilon-target', default=16, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--norm-source', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--norm-target', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--randomize', default=0, type=int)
    parser.add_argument('--logits-var', action='store_true')
    return parser.parse_args()


mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu)/std

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
    pgd_alpha = (args.pgd_alpha / 255.)
    
    act = activations.__dict__[args.activation]()
    source_var = []
    target_var = []
    base_dir = args.models_dir
    with torch.no_grad():
        if args.randomize:
            # use randomized feature extractors instead of trained
            for i in range(args.randomize):
                if i % 20 == 0:
                    print(i)
                if args.model == 'resnet18':
                    model = resnet18(num_classes=10, activation=act)
                elif args.model == 'wrn_28_10':
                    model = wrn_28_10(num_classes=10, activation=act)
                elif args.model == 'vgg16':
                    model = vgg16_bn(activation=act, num_classes=10)
                else:
                    raise ValueError('unknown model')
                model.apply(initialize_weights)
                model = nn.DataParallel(model).cuda()
                model.eval()
                total_variation = 0
                total_variation_target = 0
                for i, batch in enumerate(test_batches):
                    X, y = batch['input'], batch['target']
                    Xvar1, Xvar2 = variation_single(X, model, epsilon_source, norm=args.norm_source, step_size=pgd_alpha, num_steps=args.attack_iters, normalize=normalize, logits=args.logits_var)
                    var_source = torch.norm(model(normalize(Xvar1), feature=(not args.logits_var)) - model(normalize(Xvar2), feature=(not args.logits_var)), dim=-1, p=2).sum().item()
                    total_variation += var_source
                    Xvar1, Xvar2 = variation_single(X, model, epsilon_target, norm=args.norm_target, step_size=pgd_alpha, num_steps=args.attack_iters, normalize=normalize, logits=args.logits_var)
                    var_target = torch.norm(model(normalize(Xvar1), feature=(not args.logits_var)) - model(normalize(Xvar2), feature=(not args.logits_var)), dim=-1, p=2).sum().item()
                    total_variation_target += var_target
                        #print(var_source, var_target)
                total_variation /= len(test_set)
                total_variation_target /= len(test_set)
                source_var.append(total_variation)
                target_var.append(total_variation_target)
        
        else:
            for folder in os.listdir(base_dir):
                if args.norm_source == 'l_2':
                    if 'l2' not in folder:
                        continue
                if args.norm_source == 'l_inf':
                    if 'l2' in folder:
                        continue
                path = os.path.join(base_dir, folder)
                for f in os.listdir(path):
                    if 'model' in f:
                        f = os.path.join(path, f)
                        print('Evaluating {}'.format(f))
                        if args.model == 'resnet18':
                            model = resnet18(num_classes=10, activation=act)
                        elif args.model == 'wrn_28_10':
                            model = wrn_28_10(num_classes=10, activation=act)
                        elif args.model == 'vgg16':
                            model = vgg16_bn(activation=act, num_classes=10)
                        else:
                            raise ValueError('unknown model')
                        model.load_state_dict(filter_state_dict(torch.load(f)))
                        model = nn.DataParallel(model).cuda()
                        model.eval()
                        total_variation = 0 
                        total_variation_target = 0
                        for i, batch in enumerate(test_batches):
                            X, y = batch['input'], batch['target']
                            Xvar1, Xvar2 = variation_single(X, model, epsilon_source, norm=args.norm_source, step_size=pgd_alpha, num_steps=args.attack_iters, normalize=normalize, logits=args.logits_var)
                            var_source = torch.norm(model(normalize(Xvar1), feature=(not args.logits_var)) - model(normalize(Xvar2), feature=(not args.logits_var)), dim=-1, p=2).sum().item()
                            total_variation += var_source
                            Xvar1, Xvar2 = variation_single(X, model, epsilon_target, norm=args.norm_target, step_size=pgd_alpha, num_steps=args.attack_iters, normalize=normalize, logits=args.logits_var)
                            var_target = torch.norm(model(normalize(Xvar1), feature=(not args.logits_var)) - model(normalize(Xvar2), feature=(not args.logits_var)), dim=-1, p=2).sum().item() 
                            total_variation_target += var_target
                            #print(var_source, var_target)
                        total_variation /= len(test_set)
                        total_variation_target /= len(test_set)
                        source_var.append(total_variation)
                        target_var.append(total_variation_target)

    save_str = 'cif10_{}_{}_{}_{}_{}_'.format(args.model, args.norm_source, args.epsilon_source, args.norm_target, args.epsilon_target)
    if args.randomize:
        save_str += 'random_'
    if args.logits_var:
        save_str += 'logits_'
    np.save(save_str + 'source.npy', np.array(source_var))
    np.save(save_str + 'target.npy', np.array(target_var))

if __name__ == '__main__':
    main() 
