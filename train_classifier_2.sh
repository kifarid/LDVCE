#!/bin/bash
#PBS -N train_classifier
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16:gpus=8,mem=20gb,walltime=24:00:00
#PBS -m a
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student
source /misc/student/faridk/miniconda3/bin/activate ldm
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
#conda activate ldm
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
python -m main --base configs/latent-diffusion/classifier.yaml -t
exit 0