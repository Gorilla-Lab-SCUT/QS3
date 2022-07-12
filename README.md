# Quasi-Balanced Self-Training on Noise-Aware Synthesis of Object Point Clouds for Closing Domain Gap

### [Paper](https://arxiv.org/abs/2203.03833) | [Data](https://drive.google.com/drive/folders/1gKLKxGsvAyJo0K4BxDkgBT_r2wk1F5Vm?usp=sharing) | [Supplementary Materials]



This repository contains an implementation for the ECCV 2022 paper <a href="https://arxiv.org/abs/2203.03833"> Quasi-Balanced Self-Training on Noise-Aware Synthesis of Object Point Clouds for Closing Domain Gap</a>.

This paper introduces an integrated scheme consisting of physically realistic synthesis of object point clouds via rendering stereo images via projection of speckle patterns onto CAD models and a novel quasi-balanced self-training designed for more balanced data distribution by sparsity-driven selection of pseudo labeled samples for long tailed classes.

## Installation Requirments
The code for Mesh2Point pipeline that generates noisy point clouds is compatible with <a href="https://download.blender.org/release/">blender 2.93</a>, which also complile a default python environment. We use our Mesh2Point to scan the ModelNet dataset then get a noisy point cloud dataset--namely SpeckleNet, and the generated datasets are available <a href="https://drive.google.com/drive/folders/1gKLKxGsvAyJo0K4BxDkgBT_r2wk1F5Vm?usp=sharing">here</a> (SpeckleNet10 for 10 categories and SpeckleNet40 for 40 categories). If you want to scan your own 3D models, please download blender <a href="https://download.blender.org/release/">blender 2.93</a> and install the required python libraries in blender's python environment by running:
```
path_to_blender/blender-2.93.0-linux-x64/2.93/python/bin/pip -r install Mesh2Point_environment.yml
```

Meanwhile, we also release the code for Quasi-Balanced Self-Training (QBST), which is compatible with python <font color="#dd0000">xx</font><br /> and pytorch <font color="#dd0000">xx</font><br />.

You can create an anaconda environment called QBST with the required dependencies by running:
```
conda env create -f QBST_environment.yml
conda activate QBST
```

## Usage
### Obtain noisy data
#### Data
We use our Mesh2Point pipeline to scan <a href="https://modelnet.cs.princeton.edu/">ModelNet</a> and generate a new dataset <a href="https://drive.google.com/drive/folders/1wcERGIKvJRlCbSndM_DCSfcvjiGvv_5g?usp=sharing">SpeckleNet</a>. Note that blender cannot import ModelNet's original file format, so we convert Object File Format (.off) to Wavefront OBJ Format (.obj). The converted version ModelNet_OBJ is available <a href="https://drive.google.com/drive/folders/1gKLKxGsvAyJo0K4BxDkgBT_r2wk1F5Vm?usp=sharing">here</a>.

<p align="center">
  <img width="100%" src="./figures/Mesh2Point.png"/>
</p>

You can also scan your own 3D model dataset using:
```
CUDA_VISIBLE_DEVICES=0 path_to_blender/blender-2.93.0-linux-x64/blender ./blend_file/spot.blend -b --python scan_models.py --  --view=1 --modelnet_dir=path_to_model_dataset --category_list=bed
```
Notice that you need to organize your own data in the same architecture as ModelNet.

#### Training with ordinary model
```
To be done...
```
#### Training with QBST
```
To be done...
```
#### Evaluation on ScanNet10 and DepthScanNet10
<a href="https://drive.google.com/drive/folders/1gKLKxGsvAyJo0K4BxDkgBT_r2wk1F5Vm?usp=sharing">ScanNet10</a> is a realistic dataset generated by <a href="https://arxiv.org/abs/1911.02744">PointDAN</a>. It is extracted from a smooth mesh dataset that reconstructed from noisy depth frames. <a href="https://drive.google.com/file/d/1gWf5X8HVt0cFpDf-I-lpoqrOBc_kqVL-/view?usp=sharing">DepthScanNet10</a> is directly extracted from noisy depth frames sequence, which keep more noisy points and therefore more realistic than ScanNet10. Both two datasets use depth frames sequence from <a href="http://www.scan-net.org/">ScanNet</a>. 

For evaluating model on ScanNet10 and DepthScanNet10, running:
```
To be Done...
```

## Citation
If you find our work useful in your research, please consider citing:

        @article{chen2022quasi,
        title={Quasi-Balanced Self-Training on Noise-Aware Synthesis of Object Point Clouds for Closing Domain Gap},
        author={Chen, Yongwei and Wang, Zihao and Zou, Longkun and Chen, Ke and Jia, Kui},
        journal={arXiv preprint arXiv:2203.03833},
        year={2022}
        }

## TODO
- [ ] update scannet10 link
- [ ] update modelnet_obj link
- [ ] upload QBST code
- [ ] upload ordinary code
- [ ] generate environment.yml
- [ ] update supplental link
