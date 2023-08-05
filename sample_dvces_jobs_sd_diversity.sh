#!/bin/bash
#PBS -N LDVCE_diversity
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 4

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname

seed=$PBS_ARRAYID

python -m scripts.dvce --config-name=diversity \
    data.batch_size=4 \
    output_dir=/misc/lmbraid21/faridk/LDCE_sd_correct_seed_${seed} \
    seed=$seed \
    sampler.guidance=projected \
    sampler.classifier_lambda=3.95 \
    sampler.dist_lambda=1.2 \
    sampler.cone_projection_type=zero_binning \
    sampler.deg_cone_projection=50. > logs/LDCE_sd_correct_seed_${seed}.log

exit 0


