export CUDA_VISIBLE_DEVICES=0
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp
  # --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025.csv \
# python universal_models_fusion/multi_sensor_panopticon_4.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --batch_size 32 \
#   --epochs 50 \
#   --train_backbone \
#   --freeze_backbone_epochs 1 \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --sensor_aux_loss_weight 0.3 \
#   --num_workers 8 \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_tests5p \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "max_pooling fusion universal overlap dataset include all 2025 (schema2)" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --row_fusion_mode max    #{map,max}
  # --local_cache_warmup \
  # --local_cache_workers 16 \
  # --local_cache_min_free_gb 50


# python universal_models_fusion/multi_sensor_panopticon_4_overall_boost.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --train_backbone \
#   --batch_size 32 \
#   --epochs 50 \
#   --head_lr 1e-3 \
#   --backbone_lr 1e-4 \
#   --weight_decay 5e-4 \
#   --lr_scheduler noam \
#   --warmup_steps 4000 \
#   --row_fusion_mode map \
#   --row_gate \
#   --non_overlap_kd_weight 0.5 \
#   --gate_supervision_weight 0.1 \
#   --best_non_overlap_tolerance 0.002 \
#   --sensor_aux_loss_weight 0.3 \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_overall  \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name overall_boost_e8_kd05_gate01

# python universal_models_fusion/multi_sensor_panopticon_4_overall_boost.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --train_backbone --freeze_backbone_epochs 1 \
#   --batch_size 32 --epochs 50 \
#   --head_lr 1e-3 --backbone_lr 5e-5 --weight_decay 5e-4 \
#   --lr_scheduler noam --warmup_steps 4000 \
#   --row_fusion_mode map --row_gate \
#   --non_overlap_kd_weight 0.0 \
#   --gate_supervision_weight 0.02 \
#   --best_non_overlap_tolerance 0.002 \
#   --sensor_aux_loss_weight 0.3 \
#   --single_fused_ce_weight 0.5 \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_overall \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name overall_boost_e8_fix_single_kd0_gate002_blr5e5_freeze1

# python universal_models_fusion/multi_sensor_panopticon_4_overall_boost.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --epochs 50 \
#   --batch_size 32 \
#   --num_workers 8 \
#   --train_backbone \
#   --head_lr 1e-3 \
#   --backbone_lr 1e-4 \
#   --row_fusion_mode map \
#   --row_gate \
#   --sensor_aux_loss_weight 0.3 \
#   --non_overlap_kd_weight 0.5 \
#   --gate_supervision_weight 0.1 \
#   --single_fused_ce_weight 0.5 \
#   --single_weight_s2 1.0 \
#   --single_weight_l89 1.0 \
#   --single_weight_s5p 1.2 \
#   --single_weight_wv3 2.0 \
#   --best_non_overlap_tolerance 0.002 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name overall_boost_e8_kd05_gate01_singlew_wv3x2


# python universal_models_fusion/multi_sensor_panopticon_4.py \
#   --train_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_train_geo_resplit.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_geo_resplit.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --batch_size 32 \
#   --epochs 50 \
#   --train_backbone \
#   --freeze_backbone_epochs 1 \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --sensor_aux_loss_weight 0.3 \
#   --num_workers 8 \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_tests5p \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "Geo universal overlap dataset include all 2025 (schema2)" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --row_fusion_mode max    #{map,max}

python universal_models_fusion/multi_sensor_panopticon_4.py \
  --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid_emit_binary_mask_cleaned_train.csv \
  --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid_emit_binary_mask_cleaned_test.csv \
  --weights weights/panopticon_vitb14_teacher.pth \
  --batch_size 32 \
  --epochs 50 \
  --train_backbone \
  --freeze_backbone_epochs 1 \
  --backbone_lr 5e-5 \
  --head_lr 1e-3 \
  --sensor_aux_loss_weight 0.3 \
  --num_workers 8 \
  --device cuda \
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_tests5p \
  --use_wandb \
  --wandb_project baselines \
  --wandb_run_name "Geo universal overlap dataset include all 2025 (schema2)" \
  --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
  --local_cache_min_free_gb 200 \
  --row_fusion_mode max    #{map,max}

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --batch_size 32 --epochs 50 \
#   --head_lr 1e-3 \
#   --adapter_lr 1e-4 \
#   --training_schedule adapter_then_joint \
#   --stage_a_adapter_epochs 3 \
#   --stage_b_backbone_to_adapter_lr_ratio 0.4 \
#   --stage_b_adapter_lr 1e-4 \
#   --lr_scheduler noam --warmup_steps 4000 \
#   --sensor_aux_loss_weight 0.4 \
#   --consistency_loss_weight 0.05 \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_adapter_then_joint \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name adapter_then_joint_stageA3_ratio04



# python universal_models_fusion/multi_sensor_panopticon_4_contrastive_learning.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --batch_size 32 \
#   --epochs 50 \
#   --train_backbone \
#   --freeze_backbone_epochs 1 \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --num_workers 8 \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_tests5p \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "Fusion universal overlap dataset include all 2025 (schema2)" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --local_cache_warmup \
#   --local_cache_workers 16 \
#   --local_cache_min_free_gb 50

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_nonlinear_adapter \
#   --batch_size 24 \
#   --epochs 50 \
#   --num_workers 8 \
#   --train_backbone \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --adapter_lr_multiplier 2.0 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --adapter_dropout 0.05 \
#   --adapter_token_blocks 6 \
#   --phase1_backbone_epochs 5 \
#   --phase1_train_coverage 0.8 \
#   --phase1_sensor_coverages "s2=2.0,wv3=0.6,s5p=0.9,l89=0.4" \
#   --phase2_train_shared_adapter \
#   --sensor_aux_loss_weight 0.4 \
#   --sensor_aux_effective_num_beta 0.999 \
#   --consistency_loss_weight 0.05 \
#   --consistency_temperature 1.0 \
#   --sensor_adapter_early_stop_sensors wv3,s5p,l89,s2 \
#   --sensor_adapter_early_stopping_patience 5 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --overlap_fusion_train \
#   --overlap_head \
#   --overlap_loss_weight 0.15 \
#   --overlap_sensor_weights 's2=1.1,l89=0.9,wv3=1.0,s5p=0.6'\
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --local_cache_warmup \
#   --local_cache_workers 12 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "phase1_MHSA_sharedA_phase2_allAdapters_schema2_shared_nonlinear_private_adapter_overlap_sensor" 
  # --disable_adapter_gelu 

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_nonlinear_adapter \
#   --batch_size 24 \
#   --epochs 50 \
#   --num_workers 8 \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --adapter_lr_multiplier 2.0 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --adapter_dropout 0.05 \
#   --adapter_token_blocks 6 \
#   --phase1_backbone_epochs 0 \
#   --phase2_train_shared_adapter \
#   --sensor_adapter_early_stop_sensors "" \
#   --sensor_aux_loss_weight 0.4 \
#   --sensor_aux_effective_num_beta 0.999 \
#   --consistency_loss_weight 0.05 \
#   --consistency_temperature 1.0 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --overlap_fusion_train \
#   --overlap_head \
#   --overlap_loss_weight 0.15 \
#   --overlap_sensor_weights 's2=1.1,l89=0.9,wv3=1.0,s5p=0.6' \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --local_cache_warmup \
#   --local_cache_workers 12 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "Freeze_MHSA_train_shared_private_adapter_shared_nonlinear_private_adapter_overlap_sensor"

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --device cuda \
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor_nonlinear_adapter \
#   --batch_size 24 \
#   --epochs 50 \
#   --num_workers 8 \
#   --train_backbone \
#   --backbone_lr 5e-5 \
#   --head_lr 1e-3 \
#   --adapter_lr_multiplier 2.0 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --adapter_dropout 0.05 \
#   --adapter_token_blocks 6 \
#   --phase1_train_coverage 0.8 \
#   --phase1_sensor_coverages "s2=2.0,wv3=0.6,s5p=0.9,l89=0.4" \
#   --sensor_aux_loss_weight 0.4 \
#   --sensor_aux_effective_num_beta 0.999 \
#   --consistency_loss_weight 0.05 \
#   --consistency_temperature 1.0 \
#   --phase1_backbone_epochs 0 \
#   --phase2_train_backbone \
#   --phase2_train_shared_adapter \
#   --sensor_adapter_early_stop_sensors "" \
#   --sensor_adapter_early_stopping_patience 5 \
#   --sensor_adapter_early_stopping_warmup_epochs 5 \
#   --sensor_adapter_early_stopping_min_delta 1e-4 \
#   --fusion_group_column id \
#   --overlap_fusion logit_mean \
#   --overlap_fusion_train \
#   --overlap_head \
#   --overlap_loss_weight 0.15 \
#   --overlap_sensor_weights 's2=1.1,l89=0.9,wv3=1.0,s5p=0.6'\
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200 \
#   --local_cache_warmup \
#   --local_cache_workers 12 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name "train_all_MHSA_allAdapters_schema2_shared_nonlinear_private_adapter_overlap_sensor" 

# python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train_s5p_replaced_plus_s5p_only_old2025_s5p_balanced_by_plumeid.csv \
#   --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --weights weights/panopticon_vitb14_teacher.pth \
#   --batch_size 32 --epochs 50 --train_backbone --freeze_backbone_epochs 1 \
#   --backbone_lr 5e-5 --head_lr 1e-3 --sensor_aux_loss_weight 0.3 --num_workers 8 \
#   --device cuda --checkpoint_dir /transferdiniu2/yuyao/checkpoints/fusion_nonlinear \
#   --use_wandb --wandb_project baselines \
#   --wandb_run_name "nonlinear loraadapter fusion universal overlap dataset include all 2025 (schema2)" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
#   --local_cache_min_free_gb 200

# python universal_models_fusion/infer_overlap_or_single_models.py \
#   --csv_path /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_s5p_replaced_plus_s5p_only_old2025_with_pred_correct_filtered_overlapfixed.csv \
#   --label_column label \
#   --s2_ckpt /transferdiniu2/yuyao/checkpoints/s2/ckpt_best_test.pth \
#   --l89_ckpt /transferdiniu2/yuyao/checkpoints/l89/ckpt_best_test.pth \
#   --s5p_ckpt /transferdiniu2/yuyao/checkpoints/s5p.pth \
#   --wv3_ckpt /transferdiniu2/yuyao/checkpoints/emit.pth \
#   --batch_size 16 \
#   --sensor_sub_batch_size 2 \
#   --model_resident one_by_one \
#   --amp_dtype fp16 \
#   --num_workers 4 \
#   --use_wandb \
#   --wandb_project baselines \
#   --wandb_run_name overlap_single4_or_eval_lowmem


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
