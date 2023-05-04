#!/bin/bash
#PBS -N ldvces
#PBS -S /bin/bash
#PBS -l hostlist=chip:nodes=1:ppn=8:gpus=2,mem=16gb,walltime=24:00:00
#PBS -m a

#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 0


#3, 4, 7, 8
ulimit -n 8192
echo "changed the ulimit to 8192"
hostname
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))

python -m scripts.dvce --config-name=v7 wandb.enabled=True \
    wandb.run_id=no_cone_resnet \
    resume=True \
    data.batch_size=1 \
    data.start_sample=$PBS_ARRAYID data.end_sample=$((PBS_ARRAYID+1)) > logs/no_cone_$PBS_ARRAYID.log 

exit 0