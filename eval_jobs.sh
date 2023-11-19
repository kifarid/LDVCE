#!/bin/bash
#PBS -N eval_gen
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1,mem=15gb,walltime=6:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

##PBS -t 0

ulimit -n 8192
echo "changed the ulimit to 8192"
source /misc/student/faridk/miniconda3/bin/activate dgm-eval
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
echo "QSUB working on: ${WORKDIR}"
hostname
echo generating for $PBS_ARRAYID to $((PBS_ARRAYID+1))
#PRINT ALL CUDA RELATED ENVIRONMENT VARIABLES


nvidia-smi
nvcc --version

#Create variables to pass to python script



# Define list of output paths
output_paths=(/misc/lmbraid21/faridk/ImageNetSVCEs/ /misc/lmbraid21/faridk/ImageNetDVCEs_fixed/ /misc/lmbraid21/faridk/LDCE_w382_cc23_clsg/ /misc/lmbraid21/faridk/LDCE_w382_cc23/ /misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/)

# Loop through output paths and run script
for output_path in "${output_paths[@]}"
do
    python eval_utils/compute_fid_new.py --drop_mismatch --output-path "$output_path"
done

# don not exit, to keep the node
# loop for 10 minutes
# loop for 10 minutes

