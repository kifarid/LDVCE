#!/bin/bash
#PBS -N finetune
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M schrodi@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu

ulimit -n 8192
echo "changed the ulimit to 8192"

source ~/.bashrc
conda activate ldm
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh

WORKDIR="/home/schrodi/repos/LDVCE"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"

seed=0

python utils/finetune_resnet50.py

exit 0


