#!/bin/bash
#PBS -N eval_celeb
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=2,mem=16gb,walltime=3:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

##PBS -t 0

# run the eval_utils/evaluate_celeb_all.py script for all the celebA result directory
source ~/.bashrc
conda activate ldm_fin
bash /misc/software/cuda/add_environment_cuda11.6.2-cudnn-8.4.1-ubuntu2004.sh
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR
# Path: eval_utils/evaluate_celeb_all.py
# evaluate the celebA result directory and save the result to a csv file
base_result_path=/misc/lmbraid21/faridk/celeb_rebuttal

# for result_path in $base_result_path/celebA/*; do
#     echo $result_path
#     python eval_utils/evaluate_celeb_all.py --path $result_path --target_csv ./results/celebA/hp_evals.csv \
#         --query_label 39


python eval_utils/evaluate_celeb_all.py --path ${base_result_path}/age_032_425_345 --target_csv ./results/celebA/hp_evals_age.csv \
     --query_label 39

#python eval_utils/evaluate_celeb_all.py --path ${base_result_path}/smile_032_425_345 --target_csv ./results/celebA/hp_evals_smile.csv \
#  --query_label 31



# for result_path in $base_result_path/celeb_smile_np_*; do
#     echo "Processing $result_path"
#     python eval_utils/evaluate_celeb_all.py --path $result_path --target_csv ./results/celebA/hp_evals.csv \
#         --query_label 31
# done
