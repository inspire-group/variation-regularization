# Code for training and evaluation with non-$\ell_p$ sources

This directory is adapted from the code for Perceptual Adversarial Training (PAT) located [here](https://github.com/cassidylaidlaw/perceptual-advex).  The primary scripts used are:

- ```adv_train_var.py``` - for training models using PAT-VR
- ```adv_train_recolor_stadv.py``` - for training models with either recolor or stadv source with variation regularization
- ```evaluate_trained_model.py``` - evaluates models. Used for generating Table 2 and Table 11.

## Training and Evaluation for PAT-VR
To train a ResNet50 model using PAT-VR with $\lambda=0.1$ on CIFAR-10 with AlexNet-based LPIPS source of bound $\epsilon=1$:

```python adv_train_var.py --arch resnet50  --dataset cifar --continue --log_dir nptm_var --alexnet --var_reg 0.1 --dataset_path PATH_TO_CIFAR10_DATA --attack "FastLagrangePerceptualAttack(model, bound=1, lpips_model='alexnet_cifar', num_iterations=10)" --only_attack_correct```

- $\lambda$ is specified by ```--var-reg```
- changing "bound=1" to "bound=0.5" reproduces results for training with $\epsilon=0.5$
- models will be saved to the directory specified in ```--log_dir```

To evaluate models trained using PAT-VR (Table 8 in Appendix):

```python evaluate_trained_model.py --dataset cifar --checkpoint PATH_TO_MODEL_CHECKPOINT --dataset_path PATH_TO_CIFAR10_DATA --arch resnet50 --batch_size 256 --source "NoAttack()" "PerceptualPGDAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')" "LagrangePerceptualAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')" --attacks "AutoLinfAttack(model, 'cifar', bound=8/255)" "AutoL2Attack(model, 'cifar', bound=1)" "StAdvAttack(model, num_iterations=20)" "ReColorAdvAttack(model, num_iterations=100)"```

## Training and Evaluation for StAdv Source

To train a ResNet-18 model using StAdv source with $\lambda=1$ on CIFAR-10

```python adv_train_recolor_stadv.py --stadv --batch_size 256 --arch resnet18  --dataset cifar --num_epochs 100 --continue --var_reg 1 --log_dir stadv_var --dataset_path PATH_TO_CIFAR10_DATA --only_attack_correct```

To evaluate a ResNet-18 model trained with StAdv source (Table 7 in Appendix):
```python evaluate_trained_model.py --dataset cifar --checkpoint PATH_TO_MODEL_CHECKPOINT --dataset_path PATH_TO_CIFAR10_DATA --arch resnet18 --batch_size 256 --source "NoAttack()" "StAdvAttack(model, num_iterations=100, bound=0.03)" --attacks "AutoLinfAttack(model, 'cifar', bound=4/255)" "AutoL2Attack(model, 'cifar', bound=0.5)" "StAdvAttack(model, num_iterations=100)" "ReColorAdvAttack(model, num_iterations=100)" --union_source```

## Training and Evaluation for Recolor source

To train a ResNet-18 model using Recolor source with $\lambda=1$ on CIFAR-10

```python adv_train_recolor_stadv.py --batch_size 256 --arch resnet18  --dataset cifar --num_epochs 100 --continue --var_reg 1 --log_dir recolor_var --dataset_path PATH_TO_CIFAR10_DATA --only_attack_correct```


To evaluate a ResNet-18 model with Recolor source (Table 7 in Appendix):
```python evaluate_trained_model.py --dataset cifar --checkpoint PATH_TO_MODEL_CHECKPOINT --dataset_path PATH_TO_CIFAR10_DATA --arch resnet18 --batch_size 256 --source "NoAttack()" "ReColorAdvAttack(model, num_iterations=100, bound=0.04)" --attacks "AutoLinfAttack(model, 'cifar', bound=4/255)" "AutoL2Attack(model, 'cifar', bound=0.5)" "StAdvAttack(model, num_iterations=100)" "ReColorAdvAttack(model, num_iterations=100)" --union_source```