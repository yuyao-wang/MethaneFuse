#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

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
#     --lora_rank 128 \
#     --lora_alpha 128 \
#     --lora_dropout 0.05 \
#     --lora_targets attn.qkv,attn.proj \
#     --data_parallel \
#     --lora_warmup_epochs 2

    #  --weights weights/panopticon_vitb14_teacher.pth \

# python universal_models/multi_sensor_panopticon_lora.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s5p_patches_3x3_to_224_offl_triplet/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --sensor_switch_interval 100 \
#     --weights weights/panopticon_vitb14_teacher.pth --lora_rank 64 --lora_alpha 64 --lora_dropout 0.05 --lora_targets attn.qkv,attn.proj --data_parallel --lora_warmup_epochs 13

# python universal_models/multi_sensor_panopticon_lora.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --sensor_switch_interval 100 \
#     --weights none --lora_rank 128 --lora_alpha 128 --lora_dropout 0.05 --lora_targets attn.qkv,attn.proj --data_parallel --lora_warmup_epochs 2

# python baseline/s2_temporal_resnet34.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/test.csv \
#     --device cuda --use_wandb --wandb_project baselines --num_workers 12 

# python universal_models/multi_sensor_panopticon.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --weights none
#     --data_parallel

# python universal_models/multi_sensor_panopticon_seperate_ViTLN.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 5e-5 --head_lr 1e-4 \
#   --freeze_backbone_epochs 2 \
#   --sensor_sampling_alpha 0.85 --sensor_loss_weighting inv_sqrt \
#   --sensor_loss_weight_max 2.5 --sensor_loss_warmup_epochs 8 \
#   --adapter_last_blocks 12 --adapter_bottleneck_dim 16 --adapter_dropout 0.1 --adapter_cls_only \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_max_gb 250 \
#   --local_cache_warmup --local_cache_workers 18 

# python universal_models/multi_sensor_resnet18.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 \
#     --data_parallel --local_cache_max_gb 250

# python universal_models/multi_sensor_panopticon_seperate_ViTFFN.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 \
    # --data_parallel --local_cache_max_gb 250

python universal_models/multi_sensor_panopticon_seperate_ViTFFN_2FC.py \
    --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
    --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
    --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
    --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_workers 12 \
    --disable_checkpoints

# python examples/dino_clssifier_head_EMIT_simulated_wv3_t0_one_block.py \
#      --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/EMIT_simulated_WV3_L2A_60resolution_NOnorm/train_permian.csv \
#      --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/EMIT_simulated_WV3_L2A_60resolution_NOnorm/test_permian.csv \
#      --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --batch_size 32
    # --local_cache_max_gb 300

# # Prithvi only on S2
# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor s2 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#   --models prithvi \
#   --prithvi-repo-id ibm-nasa-geospatial/Prithvi-EO-2.0-300M

# AnySat scores on S2
# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor s2 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#   --models anysat \
#   --out-csv checkpoints/channel_scores_anysat_s2.csv

# AnySat scores on L89
# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor l89 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_2025_balanced.csv \
#   --models anysat \
#   --out-csv checkpoints/channel_scores_anysat_l89.csv

