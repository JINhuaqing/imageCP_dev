#!/bin/bash
#### The job script, run it as qsub wynton_sub.sh

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
####$ -cwd
##### set job working directory
#$ -wd /wynton/home/jianglab/hjin/MyResearch/imageCP_dev/bash_scripts/
#### Output file
#$ -o wynton/logs/Simu_$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/Simu_$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=2G
#### number of cores 
#$ -pe smp 40
#### Maximum run time 
#$ -l h_rt=48:00:00
#### job requires up to 2 GB local space
#$ -l scratch=2G
#### Specify queue
###  gpu.q for using gpu, sometimes, long.q can run GPU
###  if not gpu.q, do not need to specify it
##$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M
#### Specify job name
#$ -N wake_R

#Your script

module load CBI miniforge3
conda activate imageCP

python -u ../python_scripts/simu_gen1.py --n_jobs 30 --noise_type $1 --fmodel_type $2 --kernel_fn $3 --X_type $4

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
