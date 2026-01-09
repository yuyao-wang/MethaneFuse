import tifffile as tiff

tif_path = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/train_tryout_3/carbonmapper_data_temporal_split_classification/45555/s2.tif"
arr = tiff.imread(tif_path)

print(arr.shape)
print(arr.dtype)
