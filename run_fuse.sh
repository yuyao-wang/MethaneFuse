export CUDA_VISIBLE_DEVICES=0
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp

python universal_models_fusion/multi_sensor_panopticon_4.py \
  --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_train.csv \
  --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset/manifest_multisensor_crop_scheme2_test.csv \
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
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/multi_sensor \
  --use_wandb \
  --wandb_project baselines \
  --wandb_run_name "Fusion universal overlap dataset include all 2025 (schema2)" \
  --local_cache_dir /diniuvol/yuyao/local_train_temp_cache 
#   --local_cache_warmup \
#   --local_cache_workers 16 \
#   --local_cache_min_free_gb 50


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
