#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp


# Keep Python multiprocessing/tempfile artifacts off the root disk (/tmp).
# TMP_ROOT="/transferdiniu2/yuyao/tmp/panopticon"
# mkdir -p "${TMP_ROOT}"
# export TMPDIR="${TMP_ROOT}"
# export TMP="${TMP_ROOT}"
# export TEMP="${TMP_ROOT}"

# python examples/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/train_2024.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/test_2024.csv \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --use_wandb \
#     --wandb_project baselines \
#     --num_workers 18 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 --batch_size 16 \


# python universal_models/multi_sensor_panopticon_4.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4_geo.csv  \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4_geo.csv  \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --use_wandb \
#     --wandb_project baselines \
#     --wandb_run_name "universal baseline 4 geo" \
#     --num_workers 8 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_min_free_gb 50 \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --batch_size 32 \
#     --checkpoint_dir /transferdiniu2/yuyao/checkpoints/universal

# python universal_models/multi_sensor_panopticon_4_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /home/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 50 \
#   --local_cache_warmup \
#   --local_cache_workers 18 \
#   --epochs 50 \
#   --batch_size 32 \
#   --num_workers 8 \
#   --train_backbone \
#   --backbone_lr 1e-4 \
#   --head_lr 1e-4 \
#   --adapter_lr 2e-4 \
#   --phase1_backbone_epochs 5 \
#   --freeze_backbone_first_blocks 5 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "loraadapter_updated_2_phases_earlystopping"

# from scratch (initialize from --weights, no --resume)
# python universal_models/multi_sensor_panopticon_4_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --wandb_run_name "loraadapter_updated_2_phases_earlystopping_undersampling" \
#   --epochs 50 \
#   --batch_size 32 \
#   --num_workers 8 \
#   --train_backbone \
#   --backbone_lr 1e-4 \
#   --head_lr 1e-4 \
#   --adapter_lr 2e-4 \
#   --phase1_backbone_epochs 5 \
#   --freeze_backbone_first_blocks 999 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --use_wandb \
#   --wandb_project baselines \
#   --local_cache_dir /home/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 50 \
#   --local_cache_warmup \
#   --local_cache_workers 18 \
#   --phase1_sensor_coverages "wv3=0.55,s5p=0.25,l89=0.55"



# python universal_models/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4_geo.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4_geo.csv \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /home/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 50 \
#   --local_cache_warmup \
#   --local_cache_workers 18 \
#   --epochs 50 \
#   --batch_size 32 \
#   --num_workers 16 \
#   --train_backbone \
#   --backbone_lr 1e-4 \
#   --head_lr 1e-4 \
#   --adapter_lr 2e-4 \
#   --phase1_backbone_epochs 4 \
#   --freeze_backbone_first_blocks 5 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "nonlinear_adapter_updated_2_phases_earlystopping_undersampling_geo" \
#   --phase1_sensor_coverages "wv3=0.55,s5p=0.25,l89=0.55"

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_warmup \
#   --local_cache_workers 12 \
#   --batch_size 16 \
#   --epochs 50 \
#   --head_lr 1e-3 \
#   --backbone_lr 1e-4 \
#   --adapter_lr_multiplier 2.0 \
#   --phase1_backbone_epochs 4 \
#   --phase1_train_coverage 0.8 \
#   --phase1_sensor_coverages "wv3=0.55,s5p=0.25,l89=0.55" \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --overlap_sensor_weights "s2=1.1,l89=1.0,wv3=1.0,s5p=0.6" \
#   --overlap_fusion_train \
#   --overlap_head \
#   --overlap_loss_weight 0.15 \
#   --sensor_stats_cache /transferdiniu2/yuyao/checkpoints/multi_sensor/sensor_stats_train_4_geo.json \
#   --sensor_stats_seed 42 \
#   --num_workers 8
 # --recompute_sensor_stats --sensor_stats_max_samples_per_sensor 3000 \




# python universal_models/multi_sensor_panopticon_seperate_ViTLNFFN.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#     --device cuda \
#     --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --use_wandb \
#     --wandb_project baselines \
#     --wandb_run_name "ViTLNFFN_4" \
#     --num_workers 8 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_min_free_gb 50 \
#     --batch_size 32 \
#     --epochs 50 
    # --local_cache_warmup \
    # --local_cache_workers 18 \


# test 1
# python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter_upper_layers_4.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.55,wv3=0.55 \
#   --sensor_head_lr_mult s2=1.3,l89=0.6,s5p=0.6,wv3=0.6 \
#   --adapter_first_blocks 5 --lora_rank 16 --lora_alpha 16 \
#   --freeze_vit_in_adapter_blocks \
#   --early_stopping --early_stopping_metric composite --early_stopping_patience 6 --early_stopping_warmup_epochs 6 \
#   --sensor_threshold_min_samples 128 --sensor_threshold_ema 0.3 \
#   --train_augment --aug_noise_std 0.02 --aug_erase_prob 0.2 \
#   --use_wandb --wandb_project baselines --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache  --local_cache_warmup --local_cache_workers 18

# # test 2 (resume after disk-full interruption)
# RUN_NAME="vitln4_earlystop_20260308_182557"
# OLD_CKPT_ROOT="checkpoints/multi_sensor_earlystop"
# NEW_CKPT_ROOT="/transferdiniu2/yuyao/checkpoints/multi_sensor_earlystop"
# OLD_RUN_DIR="${OLD_CKPT_ROOT}/${RUN_NAME}"
# NEW_RUN_DIR="${NEW_CKPT_ROOT}/${RUN_NAME}"

# mkdir -p "${NEW_RUN_DIR}"

# if [ ! -f "${NEW_RUN_DIR}/ckpt_latest.pth" ]; then
#   if [ -f "${OLD_RUN_DIR}/ckpt_latest.pth" ]; then
#     cp -f "${OLD_RUN_DIR}/ckpt_latest.pth" "${NEW_RUN_DIR}/ckpt_latest.pth"
#   elif [ -f "${OLD_RUN_DIR}/ckpt_best_score.pth" ]; then
#     cp -f "${OLD_RUN_DIR}/ckpt_best_score.pth" "${NEW_RUN_DIR}/ckpt_latest.pth"
#   elif [ -f "${OLD_RUN_DIR}/ckpt_best_test.pth" ]; then
#     cp -f "${OLD_RUN_DIR}/ckpt_best_test.pth" "${NEW_RUN_DIR}/ckpt_latest.pth"
#   else
#     echo "No resumable checkpoint found under ${OLD_RUN_DIR}"
#     exit 1
#   fi
# fi

# if [ -f "${OLD_RUN_DIR}/ckpt_best_score.pth" ] && [ ! -f "${NEW_RUN_DIR}/ckpt_best_score.pth" ]; then
#   cp -f "${OLD_RUN_DIR}/ckpt_best_score.pth" "${NEW_RUN_DIR}/ckpt_best_score.pth"
# fi
# if [ -f "${OLD_RUN_DIR}/ckpt_best_test.pth" ] && [ ! -f "${NEW_RUN_DIR}/ckpt_best_test.pth" ]; then
#   cp -f "${OLD_RUN_DIR}/ckpt_best_test.pth" "${NEW_RUN_DIR}/ckpt_best_test.pth"
# fi

# python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter_upper_layers_4_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --device cuda --epochs 50 --batch_size 32 \
#   --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --adapter_lr 2e-4 \
#   --train_sensor_epoch_ratio s2=1.0,l89=0.55,s5p=0.55,wv3=0.55 \
#   --sensor_head_lr_mult s2=1.3,l89=0.6,s5p=0.6,wv3=0.6 \
#   --adapter_first_blocks 5 --lora_rank 16 --lora_alpha 16 \
#   --freeze_vit_in_adapter_blocks \
#   --early_stopping --early_stopping_metric composite \
#   --early_stopping_patience 6 --early_stopping_warmup_epochs 6 --early_stopping_min_delta 1e-4 \
#   --sensor_threshold_min_samples 128 --sensor_threshold_ema 0.3 \
#   --train_augment --aug_noise_std 0.02 --aug_erase_prob 0.2 \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name "${RUN_NAME}" \
#   --resume \
#   --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache \
#   --local_cache_warmup --local_cache_workers 18 \
#   --checkpoint_dir "${NEW_CKPT_ROOT}"

# python examples/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
#  --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/train_balanced.csv \
#  --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/emit_wv3_temporal_-90_-180_16_to_224/test_balanced.csv \
#  --device cuda --train_backbone --backbone_lr 1e-4 --head_lr 1e-4 --use_wandb --wandb_project baselines --num_workers 18 \
#  --local_cache_dir /home/yuyao/local_train_temp_cache --local_cache_warmup --local_cache_workers 18 --batch_size 16


# python universal_models/multi_sensor_panopticon_seperate_ViTLN_loraadapter_upper_layers_4.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_4.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_4.csv \
#   --device cuda \
#   --data_parallel --data_parallel_num_gpus 2 \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --epochs 40 --batch_size 86 \
#   --train_backbone --freeze_backbone_epochs 3 \
#   --backbone_lr 6e-5 --head_lr 1e-4 --adapter_lr 1.5e-4 \
#   --weight_decay 1e-3 \
#   --adapter_first_blocks 5 --adapter_train_blocks 3 \
#   --lora_rank 16 --lora_alpha 16 --adapter_dropout 0.1 \
#   --freeze_vit_in_adapter_blocks \
#   --sensor_sampling_alpha 0.85 \
#   --train_sensor_epoch_ratio s2=0.9,l89=0.8,s5p=1.2,wv3=0.8 \
#   --sensor_head_lr_mult s2=1.1,l89=0.9,s5p=1.0,wv3=0.9 \
#   --sensor_loss_weighting inv_sqrt --sensor_loss_weight_max 2.0 --sensor_loss_warmup_epochs 8 \
#   --train_augment --aug_noise_std 0.02 --aug_erase_prob 0.2 \
#   --auto_sensor_thresholds --sensor_threshold_min_samples 128 --sensor_threshold_ema 0.3 \
#   --early_stopping --early_stopping_metric composite --early_stopping_patience 6 --early_stopping_warmup_epochs 6 --early_stopping_min_delta 5e-4 \
#   --disable_save_per_sensor_best \
#   --use_wandb --wandb_project baselines \
#   --num_workers 18 \
#   --local_cache_dir /home/yuyao/local_train_temp_cache \
#   --local_cache_fallback_dir /transferdiniu2/yuyao/local_train_temp_cache \
#   --local_cache_max_gb 0 \
#   --local_cache_min_free_gb 5 \
#   --local_cache_fallback_max_gb 80 \
#   --local_cache_fallback_min_free_gb 25 
#   --local_cache_warmup \
#   --local_cache_workers 28


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


# python universal_models_fusion/multi_sensor_panopticon_4.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --batch_size 24 \
#   --epochs 50 \
#   --head_lr 1e-3 \
#   --backbone_lr 1e-4 \
#   --train_backbone \
#   --overlap_fusion logit_mean \
#   --overlap_head \
#   --overlap_loss_weight 0.15 \
#   --overlap_fusion_train \
#   --overlap_report_jsonl /transferdiniu2/yuyao/checkpoints/multi_sensor/overlap_report_wide.jsonl \
#   --num_workers 8  \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "universal overlap dataset include all 2025 (schema2)" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_warmup \
#   --local_cache_workers 16 \
#   --local_cache_min_free_gb 20

# python universal_models_fusion/dino_clssifier_head_s2_temportal_one_block.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/s2 \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --device cuda \
#   --log_interval 10 \
#   --batch_size 24 \
#   --epochs 50 \
#   --head_lr 1e-3 \
#   --backbone_lr 1e-4 \
#   --train_backbone \
#   --num_workers 8  \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "s2 include all 2025 (schema2)" \
#   --best_ckpt_path /transferdiniu2/yuyao/checkpoints/s2/ckpt_best_test.pth

# python universal_models_fusion/dino_clssifier_head_l89_temportal_one_block.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --device cuda \
#   --train_backbone \
#   --backbone_lr 5e-5 \
#   --head_lr 5e-4 \
#   --max_grad_norm 0.5 \
#   --disable_amp \
#   --nan_fill_value 0.0 \
#   --input_clip_abs 30 \
#   --log_interval 100 \
#   --batch_size 16 \
#   --num_workers 8 \
#   --use_wandb \
#   --lr_scheduler none \
#   --wandb_project baselines \
#   --wandb_run_name "l89 include all 2025 (schema2)" \
#   --best_ckpt_path /transferdiniu2/yuyao/checkpoints/l89/ckpt_best_test.pth
#   /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_rm_wrong_s5p_prob_0p4_0p7.csv
# balanced 64
# python universal_models_fusion/dino_classifier_head_s5p_temporal_one_block.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_rm_wrong_s5p_prob_0p4_0p7.csv \
#   --device cuda \
#   --train_backbone \
#   --backbone_lr 5e-5 \
#   --head_lr 5e-4 \
#   --max_grad_norm 0.5 \
#   --log_interval 100 \
#   --batch_size 32 \
#   --num_workers 8 \
#   --use_wandb \
#   --lr_scheduler none \
#   --wandb_project baselines \
#   --wandb_run_name "s5p include all 2025 (schema2)" \
#   --best_ckpt_path /transferdiniu2/yuyao/checkpoints/s5p_all_balanced \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --weights weights/panopticon_vitb14_teacher.pth 

# python universal_models_fusion/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --device cuda \
#   --train_backbone \
#   --backbone_lr 5e-5 \
#   --head_lr 5e-4 \
#   --max_grad_norm 0.5 \
#   --log_interval 100 \
#   --batch_size 16 \
#   --num_workers 8 \
#   --use_wandb \
#   --lr_scheduler none \
#   --wandb_project baselines \
#   --wandb_run_name "EMIT include all 2025 (schema2)" \
#   --best_ckpt_path /transferdiniu2/yuyao/checkpoints/emit \
#   --weights weights/panopticon_vitb14_teacher.pth


# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_warmup --local_cache_workers 16 \
#   --device cuda \
#   --batch_size 24 --epochs 50 \
#   --train_backbone \
#   --head_lr 1e-3 --backbone_lr 1e-4 --adapter_lr 2e-4 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --disable_overlap_head \
#   --overlap_loss_weight 0 \
#   --num_workers 8 \
#   --log_interval 20 \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name "nonlinear adapter universal schema2 (no overlap loss)" \
#   --phase1_backbone_epochs 4 \
#   --phase1_train_coverage 0.8 \
#   --phase1_sensor_coverages "wv3=0.9,s5p=1,l89=0.9,s2=1" \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \

# python universal_models_fusion/multi_sensor_panopticon_4_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
#   --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_warmup --local_cache_workers 16 \
#   --device cuda \
#   --batch_size 24 --epochs 50 \
#   --train_backbone \
#   --head_lr 1e-3 --backbone_lr 1e-4 --adapter_lr 2e-4 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --disable_overlap_head \
#   --overlap_loss_weight 0 \
#   --num_workers 8 \
#   --log_interval 20 \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name "lora adapter universal schema2 (no overlap loss)" \
#   --phase1_backbone_epochs 4 \
#   --phase1_train_coverage 0.8 \
#   --phase1_sensor_coverages "wv3=0.9,s5p=1,l89=0.9,s2=1" \
#   --sensor_adapter_early_stop_sensors wv3,s5p,s2,l89 \
#   --sensor_adapter_early_stopping_patience 6 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \


