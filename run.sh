#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python examples/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
    --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/train_2024.csv \
    --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/test_2024.csv \
    --device cuda \
    --train_backbone \
    --backbone_lr 1e-4 \
    --head_lr 1e-4 \
    --use_wandb \
    --wandb_project baselines \
    --num_workers 18 \
    --local_cache_dir /home/yuyao/local_train_temp_cache \
    --local_cache_warmup \
    --local_cache_workers 18 --batch_size 16

# python universal_models/multi_sensor_panopticon_4.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --use_wandb \
#     --wandb_project baselines \
#     --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18

python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter_upper_layers_4.py \
  --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
  --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
  --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
  --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7,wv3=1.0 \
  --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9, wv3=1.0 \
  --adapter_first_blocks 5 --lora_rank 16 --lora_alpha 16 \
  --freeze_vit_in_adapter_blocks --use_wandb --wandb_project baselines

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
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18  \
#     --summary_head --summary_hidden_dim 128 --summary_dropout 0.1 --summary_loss_weight 1.0

# python universal_models/multi_sensor_panopticon_seperate_ViTLN.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 \
#     --adapter_last_blocks 5

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7 \
#   --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9 \
#   --adapter_last_blocks 5 --adapter_bottleneck_dim 16 \
#   --train_backbone \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7 \
#   --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9 \
#   --adapter_last_blocks 5 --adapter_bottleneck_dim 16 \
#   --train_backbone \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7 \
#   --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9 \
#   --adapter_last_blocks 5 --lora_rank 16 --lora_alpha 16 \
#   --freeze_vit_in_adapter_blocks --use_wandb --wandb_project baselines

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter_upper_layers.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7 \
#   --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9 \
#   --adapter_first_blocks 5 --lora_rank 16 --lora_alpha 16 \
#   --freeze_vit_in_adapter_blocks --use_wandb --wandb_project baselines

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter_upper_layers.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --weights weights/panopticon_vitb14_teacher.pth \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#     --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.7 \
#     --sensor_head_lr_mult s2=1.2,l89=0.6,s5p=0.9 \
#     --adapter_first_blocks 5 --adapter_bottleneck_dim 16 \
#     --train_backbone \
#     --use_wandb --wandb_project baselines
        # --freeze_vit_in_adapter_blocks \

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_tinyadapter.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 5e-5 --head_lr 1e-4 \
#   --sensor_sampling_alpha 0.85 --sensor_loss_weighting inv_sqrt \
#   --sensor_loss_weight_max 2.5 --sensor_loss_warmup_epochs 8 \
#   --adapter_last_blocks 7 --adapter_bottleneck_dim 16 --adapter_dropout 0.1 --adapter_cls_only \
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

# python universal_models/multi_sensor_panopticon_seperate_ViTFFN_2FC.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#     --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_workers 12 \
#     --disable_checkpoints

# python examples/dino_clssifier_head_EMIT_simulated_wv3_t0_one_block.py \
#      --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/EMIT_simulated_WV3_L2A_60resolution_NOnorm/train_permian.csv \
#      --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/EMIT_simulated_WV3_L2A_60resolution_NOnorm/test_permian.csv \
#      --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --batch_size 32
    # --local_cache_max_gb 300

# python universal_models/multi_sensor_panopticon_seperate_ViTLNFFN.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 \
#   --disable_oversample_minority \
#   --prefetch_factor 6 --persistent_workers --non_blocking_transfer \
#   --enable_tf32 --cudnn_benchmark --matmul_precision medium \
#   --fused_optimizer

# python universal_models/multi_sensor_panopticon_seperate_ViTLNFFN.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 \
#   --disable_fused_optimizer --disable_amp --disable_tf32 --disable_cudnn_benchmark \
#   --matmul_precision high --disable_oversample_minority --head_lr 3e-4 \
#     --train_sensor_epoch_ratio "s2=1.0,l89=0.55,s5p=0.7" \
#     --sensor_loss_weights "s2=1.0,l89=0.6,s5p=0.9" \
#     --summary_head --summary_loss_weight 0.7

# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor s2 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#   --models prithvi \
#   --prithvi-repo-id ibm-nasa-geospatial/Prithvi-EO-2.0-600M

# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor l89 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_2025_balanced.csv \
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
#   --anysat-patch-size 100

# # S2 + SatMAE
# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor s2 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#   --models satmae \
#   --satmae-checkpoint-url https://zenodo.org/record/7338613/files/pretrain-vit-large-e199.pth \
#   --satmae-bands B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12 \
#   --satmae-in-chans 12


# L89 + same SatMAE checkpoint (missing bands zero-filled)
# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor l89 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/data_dir_l89_L2SR/l89_temporal_16_resized_to_224_CRSfixed/test_2025_balanced.csv \
#   --models satmae \
#   --satmae-checkpoint-url https://zenodo.org/record/7338613/files/finetune-vit-base-e7.pth \
#   --satmae-bands B1,B2,B3,B4,B5,B6,B7 \
#   --satmae-fill-missing-zero \
#   --satmae-in-chans 7

# python examples/channel_scores_panopticon_prithvi.py \
#   --sensor s2 \
#   --csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv \
#   --models earthpt \
#   --earthpt-bands B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12



