#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python universal_models/multi_sensor_panopticon_lora.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/train_2025_balanced.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_filtered_2025.csv \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --use_wandb \
#     --wandb_project baselines \
#     --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --sensor_switch_interval 100 \
#     --weights none \
#     --lora_rank 64 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --lora_targets attn.qkv,attn.proj \
#     --data_parallel \
#     --lora_warmup_epochs 30

    #  --weights weights/panopticon_vitb14_teacher.pth \

# python universal_models/multi_sensor_panopticon_lora.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --sensor_switch_interval 100 \
#     --weights weights/panopticon_vitb14_teacher.pth --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05 --lora_targets attn.qkv,attn.proj --data_parallel --lora_warmup_epochs 13

python universal_models/multi_sensor_panopticon_lora.py \
    --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
    --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
    --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
    --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --sensor_switch_interval 100 \
    --weights none --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05 --lora_targets attn.qkv,attn.proj --data_parallel --lora_warmup_epochs 20