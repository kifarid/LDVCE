#!/bin/bash
#PBS -N LDVCE_multiple_classifiers_SD
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1:nvidiaRTX3090,mem=15gb,walltime=24:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q default-cpu
#PBS -t 2

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for ${classifiers_list[$PBS_ARRAYID]}

seed=3
classifiers_list=("inception_v3" "vit_b_32" "efficientnet_b7" "convnext_base")

python -m scripts.dvce --config-name=multiple_classifiers \
    data.batch_size=4 \
    output_dir=/misc/lmbraid21/faridk/LDCE_sd_${classifiers_list[$PBS_ARRAYID]}_seed_${seed} \
    sampler.guidance=projected \
    seed=$seed \
    sampler.classifier_lambda=3.4 \
    sampler.dist_lambda=1.2 \
    sampler.cone_projection_type=binning \
    classifier_model.name=${classifiers_list[$PBS_ARRAYID]} \
    sampler.deg_cone_projection=45. > logs/LDCE_sd_${classifiers_list[$PBS_ARRAYID]}_seed_${seed}.log

exit 0


