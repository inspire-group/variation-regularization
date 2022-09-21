import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def variation_single(x, model, epsilon, norm, step_size=0.01, is_random=True, num_steps=10, clip_min=0, clip_max=1, normalize=None, logits=False, var_norm='l_2'):
    if normalize is None:
        normalize = lambda x: x
    if norm == 'l_inf':
        if is_random:
            random_noise_1 = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .cuda()
                .detach()
            )
            random_noise_2 = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .cuda()
                .detach()
            )
        x_pgd = Variable(x.detach().data + random_noise_1, requires_grad=True)
        x2_pgd = Variable(x.detach().data + random_noise_2, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                if not logits:
                    loss = torch.norm(model(normalize(x_pgd), feature=True)- model(normalize(x2_pgd), feature=True), float(var_norm.split('_')[1]), dim=-1).mean()
                else:
                    loss = torch.norm(model(normalize(x_pgd))- model(normalize(x2_pgd)), float(var_norm.split('_')[1]), dim=-1).mean()
                loss.backward()
                grad1 = x_pgd.grad.detach()
                grad2 = x2_pgd.grad.detach()
            x_pgd.data = x_pgd.data + step_size * grad1.data.sign()
            x2_pgd.data = x2_pgd.data + step_size * grad2.data.sign()
            
            eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
            eta2 = torch.clamp(x2_pgd.data - x.data, -epsilon, epsilon)
            
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
            x2_pgd.data = torch.clamp(x.data + eta2, clip_min, clip_max)
        return x_pgd.detach(), x2_pgd.detach()

    if norm == 'l_2':
        if is_random:
            random_noise_1 = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).cuda().detach()
            )
            random_noise_1.renorm_(p=2, dim=0, maxnorm=epsilon)

            random_noise_2 = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).cuda().detach()
            )
            random_noise_2.renorm_(p=2, dim=0, maxnorm=epsilon)
        
        x_pgd = Variable(x.detach().data + random_noise_1, requires_grad=True)
        x2_pgd = Variable(x.detach().data + random_noise_2, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                if not logits:
                    loss = torch.norm(model(normalize(x_pgd), feature=True)- model(normalize(x2_pgd), feature=True), float(var_norm.split('_')[1]), dim=-1).mean()
                else:
                    loss = torch.norm(model(normalize(x_pgd))- model(normalize(x2_pgd)), float(var_norm.split('_')[1]), dim=-1).mean()
                loss.backward()
                grad1 = x_pgd.grad.detach()
                grad2 = x2_pgd.grad.detach()
            
            # renorming gradient
            grad_norms = grad1.view(len(x), -1).norm(p=2, dim=1)
            grad2_norms = grad2.view(len(x), -1).norm(p=2, dim=1)
            
            grad1.div_(grad_norms.view(-1, 1, 1, 1))
            grad2.div_(grad2_norms.view(-1,1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad1[grad_norms == 0] = torch.randn_like(
                    grad1[grad_norms == 0]
                )
            if (grad2_norms == 0).any():
                grad2[grad2_norms == 0] = torch.randn_like(
                    grad2[grad2_norms == 0]
                )
            
            x_pgd.data += step_size * grad1.data
            eta = x_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)

            x2_pgd.data += step_size * grad2.data
            eta = x2_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x2_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)

        return x_pgd.detach(), x2_pgd.detach()
            
    if norm == 'l_1':
        pass

def upper_var_single(x, model, epsilon, norm, step_size=0.01, is_random=True, num_steps=10, clip_min=0, clip_max=1, normalize=None, logits=False):
    if normalize is None:
        normalize = lambda x: x
    if norm == 'l_inf':
        if is_random:
            random_noise_1 = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .cuda()
                .detach()
            )
        x_pgd = Variable(x.detach().data + random_noise_1, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = torch.norm(model(normalize(x_pgd), feature=(not logits))- model(normalize(x), feature=(not logits)), float(var_norm.split('_')[1]))
                loss.backward()
                grad1 = x_pgd.grad.detach()
            x_pgd.data = x_pgd.data + step_size * grad1.data.sign()
            
            eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
            
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
        return x_pgd.detach()

    if norm == 'l_2':
        if is_random:
            random_noise_1 = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).cuda().detach()
            )
            random_noise_1.renorm_(p=2, dim=0, maxnorm=epsilon)
        
        x_pgd = Variable(x.detach().data + random_noise_1, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = torch.norm(model(normalize(x_pgd), feature=(not logits))- model(normalize(x), feature=(not logits)), float(var_norm.split('_')[1]))
                loss.backward()
                grad1 = x_pgd.grad.detach()
            
            # renorming gradient
            grad_norms = grad1.view(len(x), -1).norm(p=2, dim=1)
            
            grad1.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad1[grad_norms == 0] = torch.randn_like(
                    grad1[grad_norms == 0]
                )
            
            x_pgd.data += step_size * grad1.data
            eta = x_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)

        return x_pgd.detach()
            
    if norm == 'l_1':
        pass

