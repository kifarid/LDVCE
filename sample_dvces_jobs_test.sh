#!/bin/bash
#PBS -N ldvces
#PBS -S /bin/bash
#PBS -l hostlist=[frieda, dicky, track]:nodes=1:ppn=4:gpus=1,mem=8gb,walltime=00:10:00
#PBS -m a

#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu


ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
#conda activate ldm
WORKDIR="/misc/student/faridk/stable-diffusion"

export NUMBER=4

cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
echo generating for 0 to 1
python -m scripts.dvce --config-name=v7 \
    data.start_sample=0 data.end_sample=1 \
    data.batch_size=2 \
    resume=True \
    wandb.run_id=bngarab wandb.enabled=True > logs/bngarab.log 

exit 0