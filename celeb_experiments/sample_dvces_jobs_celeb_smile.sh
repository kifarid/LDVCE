#!/bin/bash
#PBS -N celeb_smile_lora15
#PBS -S /bin/bash
#PBS -l nodes=terenece:ppn=8:gpus=1,mem=15gb,walltime=24:00:00 
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
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

ddim_steps=500
strength=0.32
#strength=$(echo "scale=3; 0$strength" | bc)

# Get the index corresponding to $PBS_ARRAYID
echo "Selected strength: $strength"

python -m scripts.dvce --config-name=v8_celebAHQ \
    data.batch_size=4 \
    diffusion_model.ckpt_path=/misc/student/faridk/stable-diffusion/sd15_lora_celeb_8.ckpt \
    data.query_label=31 \
    sampler.classifier_lambda=4.0 \
    sampler.dist_lambda=3.5 \
    data.num_shards=7 \
    strength=$strength \
    sampler.deg_cone_projection=90. \
    data.shard=${PBS_ARRAYID} \
    ddim_steps=${ddim_steps} \
    output_dir=/misc/lmbraid21/faridk/celeb_rebuttal/smile_sd15 > logs/celeb_smile_15_$PBS_ARRAYID.log 
exit 0


