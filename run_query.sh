export CUDA_VISIBLE_DEVICES=0
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp

DATA_ROOT=/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/legacy_param_480m
TRAIN_CSV="$DATA_ROOT/manifest_time_train.csv"
TEST_CSV="$DATA_ROOT/manifest_time_test.csv"

# /home/yuyao/miniconda3/envs/panopticon/bin/python universal_models_fusion/multi_sensor_panopticon_4.py \
#   --train_csv "$TRAIN_CSV" \
#   --test_csv "$TEST_CSV" \
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
#   --checkpoint_dir /transferdiniu2/yuyao/checkpoints/universal_480m \
#   --use_wandb \
#   --wandb_project query_dataset \
#   --wandb_run_name "query dataset 480m" \
#   --local_cache_dir /diniuvol/yuyao/local_train_temp_cache_480m \
#   --local_cache_min_free_gb 100 \
#   --row_fusion_mode max \
#   --local_cache_warmup \
#   --local_cache_workers 16 

# /home/yuyao/miniconda3/envs/panopticon/bin/python - <<'PY'
# import torch
# src = "/transferdiniu2/yuyao/checkpoints/120m_single4_retrain_20260417_201937/s2/ckpt_best_test.pth"
# dst = "/tmp/query120m_backbone_from_best.pth"
# ck = torch.load(src, map_location="cpu")
# state = ck["model"]
# if any(k.startswith("module.") for k in state):
#     state = {k[7:]: v for k, v in state.items()}
# backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
# torch.save({"backbone": backbone}, dst)
# print("saved:", dst, "num_keys:", len(backbone))
# PY

# # Direct adapter-centric phase from epoch 1:
# # - skip backbone phase by setting phase1_backbone_epochs=0
# # - keep backbone frozen in phase2 (default), and train adapters (plus heads/patch embeds)
# # Weighting from recent multi_sensor_panopticon_4.py overfit pattern:
# #   overfit drop (peak->final acc): L89=0.047, WV3=0.040, S5P=0.016, S2=0.011
# # Strategy:
# #   - downweight faster-overfitting sensors (l89/wv3)
# #   - upweight S2 to prioritize retention of the 0.8839 S2-only baseline
# #   - disable minority oversampling to avoid repeated exposure amplification
# #   - include l89 in sensor-adapter early-stop and shorten patience
# /home/yuyao/miniconda3/envs/panopticon/bin/python universal_models_fusion/multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping.py \
#   --train_csv "$TRAIN_CSV" \
#   --test_csv "$TEST_CSV" \
#   --weights "/tmp/query120m_backbone_from_best.pth" \
#   --training_schedule legacy \
#   --phase1_backbone_epochs 0 \
#   --freeze_backbone \
#   --phase2_train_shared_adapter \
#   --disable_oversample_minority \
#   --sensor_aux_weights "s2=1.30,l89=0.55,wv3=0.60,s5p=0.80" \
#   --sensor_adapter_early_stop_sensors "l89,wv3,s5p,s2" \
#   --sensor_adapter_early_stopping_warmup_epochs 3 \
#   --sensor_adapter_early_stopping_patience 3 \
#   --sensor_adapter_early_stopping_min_delta 0.0005 \
#   --overlap_sensor_weights "s2=1.5,l89=0.6,wv3=0.7,s5p=0.7" \
#   --checkpoint_dir "/transferdiniu2/yuyao/checkpoints/universal_120m_adapter" \
#   --wandb_project query_dataset \
#   --wandb_run_name "query dataset 120m nonlinear lora warmstart no-stageA s2-priority-v2" \
#   --batch_size 32 \
#   --epochs 50 \
#   --head_lr 1e-3 \
#   --backbone_lr 5e-5

/home/yuyao/miniconda3/envs/panopticon/bin/python - <<'PY'
import torch
src = "/transferdiniu2/yuyao/checkpoints/universal_480m/query dataset 480m/ckpt_best_test.pth"
dst = "/tmp/query480m_backbone_for_seg.pth"
ck = torch.load(src, map_location="cpu")
state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
if any(k.startswith("module.") for k in state):
    state = {k[7:]: v for k, v in state.items()}
backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
torch.save({"backbone": backbone}, dst)
print("saved:", dst, "num_backbone_keys:", len(backbone))
PY

/home/yuyao/miniconda3/envs/panopticon/bin/python universal_models_fusion/multi_sensor_panopticon_4_segmentation.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --tasks s2,l89,emit \
  --weights "/tmp/query480m_backbone_for_seg.pth" \
  --checkpoint_dir "/transferdiniu2/yuyao/checkpoints/universal_480m_seg" \
  --wandb_project "query_dataset" \
  --wandb_run_name "query dataset 480m seg" \
  --batch_size 24 \
  --epochs 7 \
  --freeze_backbone_epochs 2 \
  --backbone_lr 5e-5 \
  --head_lr 1e-3 \
  --local_cache_dir /diniuvol/yuyao/local_train_temp_cache_480m \
  --local_cache_min_free_gb 100
