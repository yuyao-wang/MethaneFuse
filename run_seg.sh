export CUDA_VISIBLE_DEVICES=1
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp

python universal_models_fusion/unet_multisensor_baseline_iou_plus.py \
  --train_csv "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m/manifest_time_train.csv" \
  --test_csv "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m/manifest_time_test.csv" \
  --tasks s2,l89,emit \
  --epochs 7 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_workers 8 \
  --device cuda \
  --log_interval 100 \
  --checkpoint_dir /transferdiniu2/yuyao/checkpoints/unet_multisensor_baseline_iou_plus \
  --use_wandb \
  --wandb_project "query_dataset" \
  --wandb_run_name "480m_unet_baseline_iou_plus_full" \
  --local_cache_dir /diniuvol/yuyao/local_train_temp_cache_480m \
  --local_cache_min_free_gb 100 

