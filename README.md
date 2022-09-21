# Code for "Formulating Robustness Against Unforeseen Attacks"

The code for experiments performed in the paper are organized as follows:
- ```train```: this directory contains code for training models using $\ell_2$ and $\ell_\infty$ source threat models.
- ```eval```: this directory contains code for evaluating models trained using $\ell_2$ and $\ell_\infty$ source threat models.
- ```perceptual_var```: the code within this directory is used for training and evaluating models trained with non-$\ell_p$ sources corresponding to experiments in Appendix F.1.8.
- ```Unseen_toy.ipynb```: contains the code for experiments with linear models on Gaussian data located in Appendix E

Setting up:

```pip install -r requirements.txt```

For directions on running training and evaluation, please refer to the README located within each subdirectories.