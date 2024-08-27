#!/bin/bash
#### sbatch scs_sub.sh to submit the job


#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#### Num of cores required, I think I should use --cpus-per-task other than --ntasks
#SBATCH --cpus-per-task=30
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=gpu
##SBATCH --gres=gpu:2
## anahita has A100-GPU with 40GB memoryMLP
##SBATCH --nodelist=anahita
#SBATCH --output=scs/logs/%x-%j.out
#SBATCH -J simu
#SBATCH --chdir=/home/hujin/jin/MyResearch/imageCP_dev/bash_scripts/

source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"
#module load SCS/anaconda/anaconda3

conda activate imageCP
#python -u ../python_scripts/simu_gen1wx.py --n_jobs 30

for noise_type in "normal" "lognormal"; do
    for fmodel_type in "LIN" "MLP"; do
        for kernel_fn in "diff" "none"; do
            python -u ../python_scripts/simu_gen1.py --n_jobs 30 --noise_type $noise_type --fmodel_type $fmodel_type --kernel_fn $kernel_fn
            python -u ../python_scripts/simu_gen1wx.py --n_jobs 30 --noise_type $noise_type --fmodel_type $fmodel_type --kernel_fn $kernel_fn
        done
    done
done