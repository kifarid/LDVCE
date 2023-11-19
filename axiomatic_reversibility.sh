#!/bin/bash
#PBS -N reversibility
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
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

python -m scripts.dvce --config-name=reversibility \
    data.batch_size=4 \
    output_dir=/misc/lmbraid21/faridk/axiomatic/reversibility/${seed} \
    seed=$seed \
    sampler.guidance=projected \
    sampler.classifier_lambda=3.95 \
    sampler.dist_lambda=1.2 \
    sampler.cone_projection_type=zero_binning \
    sampler.deg_cone_projection=50. > logs/reversibility_${seed}.log

exit 0


