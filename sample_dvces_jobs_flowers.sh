#!/bin/bash
#PBS -N LDCE_flowers
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 6

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))

classifier_lambda_list=(0.000001 0.000001 0.000001 0.000001 2 3 4 5)
dist_lamb_list=(0.6 1.2 2.4 4.8 1. 1.2 1.8 2.4)

python -m scripts.dvce --config-name=v8_flowers\
    data.batch_size=4 \
    strength=0.5 \
    sampler.classifier_lambda=${classifier_lambda_list[$PBS_ARRAYID]} \
    sampler.dist_lambda=${dist_lamb_list[$PBS_ARRAYID]} \
    output_dir=/misc/lmbraid21/faridk/testing/ldvce_flowers_${classifier_lambda_list[$PBS_ARRAYID]}_${dist_lamb_list[$PBS_ARRAYID]} \
     > logs/testing/ldvce_flowers_${classifier_lambda_list[$PBS_ARRAYID]}_${dist_lamb_list[$PBS_ARRAYID]}.log 

exit 0