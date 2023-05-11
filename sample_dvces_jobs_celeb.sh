#!/bin/bash
#PBS -N celeb_lvce_age
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

# strength_list=(0.4 0.4 0.8)
# ddim_steps=(500 500 500)

ddim_steps=500
strength=0.4 #${strength_list[$PBS_ARRAYID]}
#strength=$(echo "scale=3; 0$strength" | bc)
# Get the index corresponding to $PBS_ARRAYID
echo "Selected strength: $strength"

python -m scripts.dvce --config-name=v8_celebAHQ \
    data.batch_size=1 \
    sampler.guidance=projected \
    sampler.cone_projection_type=binning \
    sampler.classifier_lambda=3.8 \
    sampler.dist_lambda=5.0 \
    data.num_shards=7 \
    sampler.deg_cone_projection=40. \
    data.shard=${PBS_ARRAYID} \
    ddim_steps=${ddim_steps} \
    output_dir=/misc/lmbraid21/faridk/celeb_smile_new \
    strength=$strength > logs/celeb_smile_new_$PBS_ARRAYID.log   #${ddim_steps[$PBS_ARRAYID]} \

# python -m scripts.dvce --config-name=v8_celebAHQ \
#     data.batch_size=1 \
#     ddim_steps=${ddim_steps[$PBS_ARRAYID]} \
#     output_dir=/misc/lmbraid21/faridk/LDCE_celeb_$PBS_ARRAYID \
#     strength=$strength \
#     sampler.guidance=projected \
#     sampler.cone_projection_type=binning > logs/celeb_$PBS_ARRAYID.log 

exit 0


