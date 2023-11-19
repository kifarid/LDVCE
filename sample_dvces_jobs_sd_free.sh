#!/bin/bash
#PBS -N LDVCE_SD_free
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 7

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))


python -m scripts.dvce --config-name=v8_stable_diffusion_default \
    data.batch_size=2 \
    output_dir=/misc/lmbraid21/faridk/LDCE_sd_free \
    sampler.guidance=free \
    data.start_sample=$PBS_ARRAYID data.end_sample=$((PBS_ARRAYID+1)) > logs/LDCE_sd_free_${PBS_ARRAYID}.log 

exit 0