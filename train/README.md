# Code for training models with $\ell_p$ sources

The main runnable scripts within this directory are:
- ```train_cifar10.py```
- ```train_cifar100.py```
- ```train_imagenette.py```

The ImageNette dataset is publicly available [here](https://github.com/fastai/imagenette)

Sample script for training a ResNet-18 model on CIFAR-10 using AT-VR with $\ell_{2}$ source of $\epsilon=0.5$ and regularization strength $\lambda=1$:

```python train_cifar10.py --model resnet18 --epochs 200 --var-reg-type var --norm l_2 --epsilon 128 --pgd-alpha 19.125 --var-reg 1 --data-dir PATH_TO_CIFAR10_DATA --logits-reg --fname SAVE_MODEL_PATH```

- $\lambda$ is specified by ```--var-reg```
- ```--logits_reg``` applies variation regularization at the level of the logits.  Removing this argument allows for regularization to be applied on the input of the first fully connected layer of the network
- ```--trades``` can be added to specify training with TRADES rather than PGD adversarial training
- For training with $\ell_{\infty}$ source, use ```--norm l_inf --epsilon 8 --pgd-alpha 2```

The arguments for training with cifar100 (```train_cifar100.py```) and ImageNette (```train_imagenette.py```) are generally consistent.  The only difference is in ```train_imagenette.py```, there is no ```--model``` argument and training is hardcoded to use ResNet-18.