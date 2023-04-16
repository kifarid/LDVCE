#!/bin/bash
#PBS -N dvces
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=2,mem=24gb,walltime=24:00:00
#PBS -m a

#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
#conda activate ldm
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
python -m scripts.dvce #--config-name=v2
exit 0