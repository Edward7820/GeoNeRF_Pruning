# GeoNeRF Pruning

This repository contains a PyTorch Lightning implementation of our works, utilizing a channel pruning technique to downsize the GeoNeRF model. The implementation is based on two github repositories: https://github.com/idiap/GeoNeRF and  https://github.com/zejiangh/Filter-GaP.

## Installation

To install the dependencies, in addition to PyTorch, run:

```
pip install -r requirements.txt
```

## Evaluation and Training
To reproduce our results, download pretrained weights from [here](https://drive.google.com/drive/folders/1ZtAc7VYvltcdodT_BrUrQ_4IAhz_L-Rf?usp=sharing) and put them in [pretrained_weights](./pretrained_weights) folder. Then, follow the instructions for each of the [LLFF (Real Forward-Facing)](#llff-real-forward-facing-dataset), [NeRF (Realistic Synthetic)](#nerf-realistic-synthetic-dataset), and [DTU](#dtu-dataset) datasets.

## LLFF (Real Forward-Facing) Dataset
Download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and set its path as `llff_path` in the [config_llff.txt](./configs/config_llff.txt) file.

## NeRF (Realistic Synthetic) Dataset
Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and set its path as `nerf_path` in the [config_nerf.txt](configs/config_nerf.txt) file.

## DTU Dataset
 Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) 
and replace its `Depths` directory with [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repository](https://github.com/YoYo000/MVSNet), and set `dtu_pre_path` referring to this dataset in the [config_dtu.txt](configs/config_dtu.txt) file.

Then, download the original `Rectified` images from [DTU Website](https://roboimagedata.compute.dtu.dk/?page_id=36), and set `dtu_path` in the [config_dtu.txt](configs/config_dtu.txt) file accordingly.

## Training

Run the following commands:

```shell
python run_geo_nerf.py --config configs/config_general.txt --grow_prune --grow_prune_3d
```