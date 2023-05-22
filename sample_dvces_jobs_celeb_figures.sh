#!/bin/bash
#PBS -N LDVCE_ageee
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 1

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname


ddim_steps=500
strength=0.4 #${strength_list[$PBS_ARRAYID]}


python -m scripts.dvce --config-name=v8_celebAHQ_corrected \
    data.batch_size=1 \
    sampler.classifier_lambda=5 \
    sampler.dist_lambda=4.6 \
    data.num_shards=7 \
    data.query_label=39 \
    sampler.deg_cone_projection=40. \
    ddim_steps=${ddim_steps} \
    output_dir=/misc/lmbraid21/faridk/ageee_$PBS_ARRAYID \
    strength=$strength > logs/ageee_$PBS_ARRAYID.log   #${ddim_steps[$PBS_ARRAYID]} \

exit 0


