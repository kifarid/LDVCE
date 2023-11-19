#!/bin/bash
#PBS -N finetune_lora_celeb_1gpu
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8:gpus=1,mem=16gb,walltime=7:00:00
#PBS -o logs/
#PBS -M faridk@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export HF_HOME='/misc/lmbraid21/faridk/.cache/huggingface/'
export OUTPUT_DIR="/misc/lmbraid21/faridk/lora_finetune_2"
export DATASET_DIR='/misc/lmbraid21/faridk/CelebAMask-HQ_lora/'

source /misc/student/faridk/miniconda3/bin/activate lora
WORKDIR="/misc/student/faridk/stable-diffusion"
cd $WORKDIR

echo "QSUB working on: ${WORKDIR}"
hostname

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir $DATASET_DIR \
  --dataloader_num_workers=8 \
  --resolution=256 --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=9e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --validation_prompt="A photo of a smiling celebrity, high-quality, 1024x1024" \
  --seed=42