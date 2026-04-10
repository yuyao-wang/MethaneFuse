export CUDA_VISIBLE_DEVICES=0
mkdir -p /transferdiniu2/yuyao/temp
chmod 700 /transferdiniu2/yuyao/temp
export TMPDIR=/transferdiniu2/yuyao/temp
export TMP=/transferdiniu2/yuyao/temp
export TEMP=/transferdiniu2/yuyao/temp

# python examples/dino_clssifier_head_s2_temportal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_s2_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_s2_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "S2_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 50 

# python examples/dino_clssifier_head_l89_temportal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_l89_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_l89_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "L89_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /home/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18

# python examples/dino_classifier_head_s5p_temporal_one_block.py \
#     --train_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_s5p_geo.csv \
#     --test_csv /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/test_s5p_geo.csv \
#     --wandb_project baselines \
#     --wandb_run_name "S5p_geo_split" \
#     --device cuda \
#     --train_backbone \
#     --backbone_lr 1e-4 \
#     --head_lr 1e-4 \
#     --local_cache_dir /transferdiniu2/yuyao/local_train_temp_cache \
#     --local_cache_warmup \
#     --local_cache_workers 18 \
#     --local_cache_min_free_gb 50 \
#     --batch_size 32

# /mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/datasets_mixed_training/train_wv3_geo.csv
python universal_models_fusion/dino_clssifier_head_EMIT_simulated_wv3_temporal_one_block.py \
    --train_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_train_geo_resplit.csv \
    --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_geo_resplit.csv \
    --wandb_project baselines \
    --wandb_run_name "Geo_wv3_report" \
    --device cuda \
    --train_backbone \
    --backbone_lr 1e-4 \
    --head_lr 1e-4 \
    --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
    --local_cache_warmup \
    --local_cache_workers 18 \
    --local_cache_min_free_gb 200 \
    --batch_size 32 \
    --epoch 10

python universal_models_fusion/dino_classifier_head_s5p_temporal_one_block.py \
    --train_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_train_geo_resplit.csv \
    --test_csv /home/yuyao/panopticon/manifest_multisensor_crop_scheme2_test_geo_resplit.csv \
    --wandb_project baselines \
    --wandb_run_name "Geo_s5p_report" \
    --device cuda \
    --train_backbone \
    --backbone_lr 1e-4 \
    --head_lr 1e-4 \
    --local_cache_dir /diniuvol/yuyao/local_train_temp_cache \
    --local_cache_warmup \
    --local_cache_workers 18 \
    --local_cache_min_free_gb 200 \
    --batch_size 32 \
    --epoch 10