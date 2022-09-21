import os
import numpy as np
def write_slurm(name, eps):
    base = """#!/bin/bash
#!/bin/bash
#SBATCH --job-name={}_{}         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=72:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sihuid@princeton.edu

module load anaconda3/2020.7
conda activate torch_16

python evaluate_trained_model.py --dataset cifar --arch resnet50 --checkpoint /scratch/gpfs/sihuid/nptm_var/res50_self/{}.pth --parallel 4 --batch_size 256 --output evaluation_{}_{}.csv --dataset_path /scratch/gpfs/sihuid/data/cifar10 "NoAttack()"  "PerceptualPGDAttack(model, num_iterations=40, bound={}, projection='newtons')" "LagrangePerceptualAttack(model, num_iterations=40, bound={}, projection='newtons')"
""".format(name, eps, name, name, eps, eps, eps)
    with open('eval_lpips/eval_{}_{}'.format(name, eps), "w") as f:
        f.write(base)

if __name__ == '__main__':
    if not os.path.exists('eval_lpips'):
        os.mkdir('eval_lpips')
    folder = '/scratch/gpfs/sihuid/nptm_var/res50_self/'
    for f in os.listdir(folder):
        name = f.split('.')[0]
        for eps in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            write_slurm(name, eps)

