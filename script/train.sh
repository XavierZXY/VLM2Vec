#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2
export EXP_NAME=Qwen2vl_2B.image+visdoc+video.autoresize.lora16.BS1024.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100.3L40

export WANDB_NAME=$EXP_NAME
export EXP_DIR=./logs/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb


torchrun --nproc_per_node=3 --master_port=22007 \
    --max_restarts=0 train.py \
    --lora --lora_r 16 \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --bf16 --pooling eos --normalize True --temperature 0.02 \
    --dataloader_num_workers 8 \
    --dataset_config experiments/public/train/train_image.yaml \
    --run_name $EXP_NAME \
    --output_dir $EXP_DIR \
    --grad_cache True \
    --per_device_train_batch_size 4 \
    --gc_q_chunk_size 8 --gc_p_chunk_size 8 --interleave_batch_size 4 \
    --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 5000 \
    --warmup_steps 100 --save_steps 50 --logging_steps 1 \
    --save_safetensors True --remove_unused_columns False \
    --resume_from auto \
    --report_to wandb 2>&1 | tee $EXP_DIR/train.log