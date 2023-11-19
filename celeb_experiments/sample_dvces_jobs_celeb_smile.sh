#!/bin/bash
#PBS -N celeb_smile
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00 #
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 0

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))

export TRANSFORMERS_CACHE='/misc/lmbraid21/faridk/.cache/huggingface/hub'
export TORCH_HOME='/misc/lmbraid21/faridk/.cache/torch/'
strength_list=(0.32 0.36 0.38 0.4 0.43 0.46 0.48)
# ddim_steps=(500 500 500)

ddim_steps=500
strength={strength_list[$PBS_ARRAYID]}
#strength=$(echo "scale=3; 0$strength" | bc)

# Get the index corresponding to $PBS_ARRAYID
echo "Selected strength: $strength"

python -m scripts.dvce --config-name=v8_celebAHQ \
    data.batch_size=1 \
    data.query_label=31 \
    sampler.classifier_lambda=3.95 \
    sampler.dist_lambda=2.85 \
    data.num_shards=7 \
    strength=$strength \
    sampler.deg_cone_projection=55. \
    data.shard=0 \
    ddim_steps=${ddim_steps} \
    output_dir=/misc/lmbraid21/faridk/celeb_smile_np_$strength > logs/celeb_smile_$PBS_ARRAYID.log   #${ddim_steps[$PBS_ARRAYID]} \


exit 0


