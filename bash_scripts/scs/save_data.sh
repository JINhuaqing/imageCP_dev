#!/bin/bash
#### sbatch scs_sub.sh to submit the job


#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#### Num of cores required, I think I should use --cpus-per-task other than --ntasks
#SBATCH --cpus-per-task=40
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=long
##SBATCH --gres=gpu:2
## anahita has A100-GPU with 40GB memory
##SBATCH --nodelist=anahita
#SBATCH --output=scs/logs/%x-%j.out
#SBATCH -J save_data_testfull
#SBATCH --chdir=/home/hujin/jin/MyResearch/imageCP_dev/bash_scripts/

#source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"
module load SCS/anaconda/anaconda3

conda activate imageCP
python -u ../python_scripts/save_data.py --n_jobs 30
#### You job


