#!/bin/bash
#PBS -N cub
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1,mem=24gb,walltime=24:00:00
#PBS -m a
#PBS -M schrodi@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu

ulimit -n 8192
echo "changed the ulimit to 8192"


source ~/.bashrc

bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh

conda activate ldm

WORKDIR="/home/schrodi/repos/LDVCE"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"

python -m utils.train_cub.py

exit 0