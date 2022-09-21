import perceptual_advex
    
import torch
from perceptual_advex.distances import LPIPSDistance
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features
from perceptual_advex.utilities import get_dataset_model

import torchvision
import numpy as np
import matplotlib.pyplot as plt

def save(img):
    if len(img.size()) == 4:
        img = torchvision.utils.make_grid(img, nrow=10, padding=0)
    npimg = img.detach().cpu().numpy()
    plt.figure(figsize=(18,16), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig("found.png")

dataset, model_std = get_dataset_model(
    dataset='cifar',
    arch='resnet50',
    checkpoint_fname='data/checkpoints/cifar_pgd_l2_1.pt',
)

lpips_model = get_lpips_model('alexnet_cifar')
if torch.cuda.is_available():
    lpips_model.cuda()    
lpips_distance = LPIPSDistance(lpips_model)

# use PGD optimization to find a perturbation that causes 2 different inputs to reduce LPIPS distance
def criterion(img1, img2):
    f1 = normalize_flatten_features(lpips_distance.features(img1))
    f2 = normalize_flatten_features(lpips_distance.features(img2))
    size = [64, 16, 16]
    f1 = f1[:size[0] * size[1] * size[2]]
    f2 = f2[:size[0] * size[1] * size[2]]
    return -lpips_distance(img1, img2)
#-lpips_distance(img1, img2)


def pgd(x, x2, num_steps=80, epsilon=1, step_size=.15, clip_min=0, clip_max=1):
    x_pgd = x.detach().data
    x2_pgd = x2.detach().data
    for _ in range(num_steps):
        x_pgd.requires_grad_()
        x2_pgd.requires_grad_()
        with torch.enable_grad():
            loss = criterion(x_pgd, x2_pgd)
            #print(loss)
            loss.backward()
            #grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            #grad2 = torch.autograd.grad(loss, [x2_pgd], create_graph=False)[0].detach()
            
            grad = x_pgd.grad
            grad2 = x2_pgd.grad
            
            grad_norms = grad.view(len(x), -1).norm(p=2, dim=1)
            grad2_norms = grad2.view(len(x), -1).norm(p=2, dim=1)
            
            grad.div_(grad_norms.view(-1, 1, 1, 1))
            grad2.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad[grad_norms == 0] = torch.randn_like(
                    grad[grad_norms == 0]
                )
            if (grad2_norms == 0).any():
                grad2[grad2_norms == 0] = torch.randn_like(
                    grad2[grad2_norms == 0]
                )
            
            x_pgd = x_pgd.detach() + step_size * grad.data
            x2_pgd = x2_pgd.detach() + step_size * grad2.data
            
            eta = x_pgd - x
            eta2 = x2_pgd - x2
            
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            eta2.renorm_(p=2, dim=0, maxnorm=epsilon)
            
            x_pgd = torch.clamp(x.data + eta, clip_min, clip_max)
            x2_pgd = torch.clamp(x2.data + eta, clip_min, clip_max)
            
            #print(lpips_distance(x_pgd, x2))
    return x_pgd, x2_pgd
    
# comb through CIFAR-10 and find 2 different images that have close LPIPS distance 
batch_size = 10000
_, val_loader = dataset.make_loaders(1, batch_size, only_val=True)
inputs, labels = next(iter(val_loader))

min_img1 = None
min_img2 = None
adv_1 = None
adv_2 = None
for i in range(len(inputs)):
    print("i = {}:".format(i))
    img1 = inputs[i].unsqueeze(0).cuda()
    for j in range(i+1, len(inputs)):
        if (j % 100) == 0:
            print("i = {}, j = {}:".format(i, j))
        img2 = inputs[j].unsqueeze(0).cuda()
        dist = lpips_distance(img1, img2)
        if dist > 2:
            x_adv, x_adv2 = pgd(img1, img2, num_steps=20,epsilon=2,step_size=0.2)
            lpips_dist = lpips_distance(x_adv, x_adv2)
            #print(lpips_dist)
            if lpips_dist < 1:
                min_img1 = img1
                min_img2 = img2
                adv_1 = x_adv
                adv_2 = x_adv2
                break
                
save([min_img1, adv_1, adv_2, min_img2])
