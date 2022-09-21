# Evaluation code for models trained with $\ell_p$ sources

The main code in this directory is organized as follows:
- ```eval_uar.py```: contains code for evaluating across a variety of target threat models, used for Tables 1, 5, 6, 8, 10, 12, and 13
- ```eval_lp_range.py```: contains code for running APGD with various $\ell_{\infty}$, $\ell_2$, and $\ell_1$ radii.  Used for Figure 1 in main text and Figures 8, 9, 11, and 12 in the appendix.
- ```eval_expansion.py```: contains code for computing source variation and target variation for a directory of models, used for Figure 2 in main text

To compute robust accuracy across various threat models (as in Table 1):

```python eval_uar.py --data DATASET --arch ARCHITECTURE --data_dir PATH_TO_DATA --norm NORM --log_path logs --preprocess meanstd --checkpoint PATH_TO_CHECKPOINT --union_source```

- DATASET is "CIFAR10", "CIFAR100", or "ImageNette"
- ARCHITECTURE is resnet18, wrn_28_10, or vgg16.  Note if ImageNette is being used the architecture is hardcoded to be ResNet-18
- NORM is "L2" or "Linf" and specifies which $\ell_p$ source is used.  Note that the script is hardcoded to use $\epsilon=0.5$ for the source threat model when NORM is L2 and $\epsilon=8/255$ when NORM is Linf.
- the ```--union_source``` causes the script to report accuraces on each target unioned with the source.  This argument can be removed to report accuracies without source union.
- the evaluations will be written to the logs directory in a file named ```unforeseen_evals.log```

To compute robust accuracy for various radii of $\ell_p$ attacks (as plotted in Figure 1):
```python eval_lp_range.py --data DATASET --arch ARCHITECTURE --data_dir PATH_TO_DATA --norm NORM --log_path logs --preprocess meanstd --checkpoint PATH_TO_CHECKPOINT```
- the $\epsilon$ range used for $\ell_{\infty}$ attacks is [8/255, 25/255) with increments of 1/255
- the $\epsilon$ range used for $\ell_{2}$ attacks is [0.5, 2.3) with increments of 0.1
- the $\epsilon$ range used for $\ell_{1}$ attacks is [5, 23) with increments of 1

The output will be written to the logs directory in the npy files:
- ```linf_errs.npy```, ```l2_errs.npy```, ```l1_errs.npy```- contain numpy arrays with the computed cross entropy error for each $\epsilon$ in the specified range
- ```linf_accs.npy```, ```l2_accs.npy```, ```l1_accs.npy```- contain numpy arrays with the computed robust accuracy for each $\epsilon$ in the specified range

To compute source and target variations for plotting expansion function (currently only CIFAR-10 is supported):

```python eval_expansion.py --model ARCHITECTURE --data-dir PATH_TO_CIFAR10_DATA --epsilon-source EPSILON_SOURCE --norm-source NORM_SOURCE --epsilon-target EPSILON_TARGET --norm-target NORM_TARGET --models-dir PATH_TO_MODELS --logits-var```

- EPSILON_SOURCE, NORM_SOURCE- specify the source threat model. Similarily EPSILON_TARGET and NORM_TARGET specify the target threat model.  Here the norm argument is either "l_inf" or "l_2"
- PATH_TO_MODELS specifies a path to a directory of models that we want to compute source and target variation for.  Within this directory are folders of checkpoints that are obtained through training.  Each folder contains "l2" in the name if it was trained using $\ell_2$ source.  Otherwise the code assumes that it was trained using $\ell_{\infty}$ source.
- the code will save source and target variations into npy files located in the same directory as the code.  The name of the source variation file will be of the form ```cif10_..._source.npy``` and the target variation file will be of the form ```cif10_..._target.npy```