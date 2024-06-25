#!/bin/bash
#### sbatch scs_sub.sh to submit the job

#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#### Num of cores required, I think I should use --cpus-per-task other than --ntasks
#SBATCH --cpus-per-task=1
#####SBATCH --ntasks=30
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=gpu
#### Allocate 1 GPU resource for this job. 
#SBATCH --nodelist=titan
#SBATCH --output=prefix-%x-%j.out
#SBATCH -J test

nvidia-smi
source ~/.bashrc
source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh

conda activate gpu
python test.py
#### You job
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"


