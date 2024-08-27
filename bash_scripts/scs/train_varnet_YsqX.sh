#!/bin/bash
#### sbatch scs_sub.sh to submit the job

#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#### Num of cores required, I think I should use --cpus-per-task other than --ntasks
#SBATCH --cpus-per-task=5
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
## anahita has A100-GPU with 40GB memory
#SBATCH --nodelist=anahita
#SBATCH --output=scs/logs/%x-%j.out
#SBATCH -J train_varnet_YsqX
#SBATCH --chdir=/home/hujin/jin/MyResearch/imageCP_dev/bash_scripts/

nvidia-smi
source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh

conda activate imageCP
python
python -u ../python_scripts/train_varnet_YsqX.py --gpus 2
#### You job
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"


