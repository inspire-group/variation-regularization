from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import json
from collections import OrderedDict
import importlib
import copy 

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from perceptual_advex.distances import LPIPSDistance
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features


from models.classifier import LinearClassifier, Classifier, Classifier2, Classifier3

def train(lpips_dist, classifier, dataloader, criterion, optimizer, scheduler, gauss_aug=True, std=1):
    classifier.train()
    correct = 0
    total_imgs = 0
    avg_loss = 0
    for i, (images, target) in enumerate(dataloader):
        images, target = images.cuda(), target.cuda()
        
        xs = normalize_flatten_features(lpips_dist.features(images))
        #print(xs.size())
        
        if gauss_aug:
            xs = xs + std * torch.randn_like(xs)
        
        optimizer.zero_grad()
        
        output = classifier(xs)
        loss = criterion(output, target)
                
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        avg_loss += loss.item() / len(dataloader)

        # measure accuracy and record loss
        prediction = torch.max(output, 1) 
        correct += (prediction[1] == target).sum().item()
        total_imgs += len(target)

        
    return correct / total_imgs, avg_loss


def test(lpips_dist, classifier, dataloader):
    classifier.eval()
    correct = 0
    total_imgs = 0
    for i, (images, target) in enumerate(dataloader):
        with torch.no_grad():
            images, target = images.cuda(), target.cuda()
            xs = normalize_flatten_features(lpips_dist.features(images))
        
            output = classifier(xs)
            # measure accuracy and record loss
            prediction = torch.max(output, 1) 
            correct += (prediction[1] == target).sum().item()
            total_imgs += len(target)

    return correct / total_imgs


def main():
    parser = argparse.ArgumentParser(description="Perceptual Model Training")
    parser.add_argument(
        "--results-dir", type=str, default="results",
    )
    parser.add_argument("--data-dir", type=str, default='data_dir')
    parser.add_argument("--exp-name", type=str, default="classifier_train")
    parser.add_argument("--perceptual-ckpt", type=str, default="alexnet_best.pth.tar")
    parser.add_argument("--gauss-aug", action='store_true')
    parser.add_argument("--gauss-var", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    
    args = parser.parse_args()
    
    # create results dir (for logs, checkpoints, etc.)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(args.results_dir, "setup.log"), "a")
    )
    logger.info(args)
    
    # create model + optimizer
    lpips_model = get_lpips_model('alexnet_cifar')
    lpips_model.cuda()
    lpips_distance = LPIPSDistance(lpips_model)
    classifier = Classifier(num_classes=10).cuda().train()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_test = 0
    
    # dataloaders 
    trtrain = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor()]
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor()]
    transform_test = transforms.Compose(trval)
    
    trainset = datasets.CIFAR10(
            root=os.path.join(args.data_dir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(args.data_dir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
            
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    num_batches = len(train_loader)
    criterion = nn.CrossEntropyLoss()
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * num_batches, eta_min=0.001)
    
    # Let's roll
    for epoch in range(args.epochs):
        
        
        corr_train, loss_train = train(lpips_distance, classifier, train_loader, criterion, optimizer, lr_scheduler, args.gauss_aug, np.sqrt(args.gauss_var))
        corr_test = test(lpips_distance, classifier, test_loader)
        is_best = corr_test > best_test
        best_test = max(corr_test, best_test)
        
        if is_best:

            d = {
                "epoch": epoch + 1,
                "state_dict": classifier.state_dict(),
                "acc": best_test,
                "optimizer": optimizer.state_dict(),
            }  
            torch.save(d, 'classifier.pth.tar')
            
        logger.info("Epoch {}, Train acc {}, Train loss {}, Test acc {}".format(epoch, corr_train, loss_train, corr_test))


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
