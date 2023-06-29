#!/bin/bash

cp -rf ./data/dataset_info.json /root/workspace_law/data_ChatGLM

export CUDA_VISIBLE_DEVICES=0
python3 ./src/train_sft.py \
    --model_name_or_path /root/workspace_law/THUDM-chatglm2-6b \
    --use_v2 \
    --do_train \
    --dataset 52k,92k \
    --dataset_dir /root/workspace_law/data_ChatGLM \
    --finetuning_type lora \
    --output_dir /root/workspace_law/data_ChatGLM/output \
    --overwrite_cache \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --num_train_epochs 4 \
    --fp16 


# python3 src/cli_demo.py \
#     --model_name_or_path /root/workspace_law/THUDM-chatglm2-6b \
#     --use_v2 \
#     --checkpoint_dir /root/workspace_law/data_ChatGLM/output_export \
    
    


