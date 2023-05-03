#!/bin/bash
#PBS -N ldvces
#PBS -S /bin/bash
#PBS -l hostlist=^[dicky,track,frieda]:nodes=1:ppn=8:gpus=1,mem=16gb,walltime=24:00:00
#PBS -m a

#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 0, 1, 3, 4, 5, 6, 7, 8, 9

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))


exit 0