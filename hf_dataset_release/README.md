---
license: cc-by-nc-4.0
task_categories:
- image-classification
- image-segmentation
tags:
- methane
- remote-sensing
- multi-sensor
pretty_name: MethaneUnion
---

# MethaneUnion

This dataset product is organized as:

```text
datasets/
  temporal_split/
    original_scale/{train,test}.csv
    120m_GSD/{train,test}.csv
    360m_GSD/{train,test}.csv
    480m_GSD/{train,test}.csv
    960m_GSD/{train,test}.csv
  geo_split/
    original_scale/{train,test}.csv
    120m_GSD/{train,test}.csv
    360m_GSD/{train,test}.csv
    480m_GSD/{train,test}.csv
    960m_GSD/{train,test}.csv
```

Each CSV contains exactly these columns: `id`, `label`, `latitude`, `longitude`, `available_sensor`, `S2_t0_path`, `S2_pre_path`, `S2_pre_pre_path`, `S2_plume_label_path`, `L89_t0_path`, `L89_pre_path`, `L89_pre_pre_path`, `L89_plume_label_path`, `EMIT_t0_path`, `EMIT_pre_path`, `EMIT_pre_pre_path`, `EMIT_plume_label_path`, `S5p_temporal_path`.

Path columns point to files inside the uploaded `dataset_part_XXX.tar.gz` archives. S5P temporal files are `.npz`; plume labels are named with sensor-specific `*_plume.tif` filenames.
