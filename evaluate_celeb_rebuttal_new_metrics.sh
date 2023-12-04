#!/bin/bash
#PBS -N eval_celeb
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=2,mem=15gb,walltime=2:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

##PBS -t 0

# run the eval_utils/evaluate_celeb_all.py script for all the celebA result directory
source ~/.bashrc
conda activate dgm-eval
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR

python eval_utils/compute_fid_new.py --sfid 

base_result_path=/misc/lmbraid21/faridk
for result_path in $base_result_path/celeb_rebuttal/*; do
    echo "Processing $result_path"
    python eval_utils/compute_fid_new.py --output-path $result_path \
        --sfid
done
