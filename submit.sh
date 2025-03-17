#!/bin/sh
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J adlcv-ex-32
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set walltime limit: hh:mm --
#BSUB -W 0:30
### -- set the email address -- 
#BSUB -u hugomn2002@gmail.com
# request 1GB of system-memory
#BSUB -R "rusage[mem=1GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

source /work3/s214734/02501/miniconda3/etc/profile.d/conda.sh
conda activate adlcv-ex-32

# here follow the commands you want to execute 
python3 nmain.py