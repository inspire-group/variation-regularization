# Code for "Formulating Robustness Against Unforeseen Attacks"

The code for experiments performed in the paper are organized as follows:
- ```train```: this directory contains code for training models using $\ell_2$ and $\ell_\infty$ source threat models.
- ```eval```: this directory contains code for evaluating models trained using $\ell_2$ and $\ell_\infty$ source threat models.
- ```perceptual_var```: the code within this directory is used for training and evaluating models trained with non-$\ell_p$ sources corresponding to experiments in Appendix F.1.8.
- ```Unseen_toy.ipynb```: contains the code for experiments with linear models on Gaussian data located in Appendix E

Setting up:

```pip install -r requirements.txt```

For directions on running training and evaluation, please refer to the README located within each subdirectories.

Pretrained models:
- For $\ell_2$ and $\ell_\infty$ sources (these should be evaluated with code in ```eval``` directory): [models](https://drive.google.com/drive/folders/1A625cN91Wpv_VzNhVgYToVKEZhErxQD1?usp=sharing)
- For PAT-VR, StAdv, and ReColor sources (these should be evaluated with code in ```perceptual_var``` directory): [models](https://drive.google.com/drive/folders/11rTDxHELefgyomEy-RZxRWLFb2EZKS3S?usp=sharing)
