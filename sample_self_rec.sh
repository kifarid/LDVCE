#!/bin/bash
#PBS -N ldvces
#PBS -S /bin/bash
#PBS -l nodes=frieda:ppn=8:gpus=1,mem=24gb,walltime=24:00:00
#PBS -m a

#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
#conda activate ldm
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
python -m scripts.dvce --config-name=v7 wandb.enabled=True \
    data.start_sample=0 data.end_sample=1 

exit 0