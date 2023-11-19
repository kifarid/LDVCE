#!/bin/bash
#PBS -N eval_gen
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1,mem=15gb,walltime=1:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

##PBS -t 0

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate dgm-eval
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))
#PRINT ALL CUDA RELATED ENVIRONMENT VARIABLES


nvidia-smi
nvcc --version


python eval_utils/compute_fid_new.py
# don not exit, to keep the node
# loop for 10 minutes
# loop for 10 minutes

