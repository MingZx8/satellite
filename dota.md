# Environment
+ Ubuntu 18.04(Linux)
+ dual GPU
  - GTX 745
  - RTX 2070

# Prerequisite
+ Python 3.6.9
+ GCC 7.3.0
+ CUDA 10.0

# Installation
###### *Not necessary: * Create and activate a virtual environment 
```
conda create -n <environment name> python=3.6
conda activate <environement name>
```
###### Other packages  
+ PyTorch 1.3.1
+ cython

#### Clone repository
```
cd <your_project_path>  
git clone https://github.com/dingjiansw101/AerialDetection.git  
cd AerialDetection
```

#### Compile cuda extentions
```
./compile.sh
```

#### Install AerialDetection
```
pip install -r requirements.txt
python setup.py develop
```

#### Install DOTA_devkit
```
sudo apt-get install swig
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

# Prepare Dataset
###### Can first use the DOTA dataset to understand the data structure and training/testing process:
#### Download the data
[DOTA Download](https://captain-whu.github.io/DOTA/dataset.html)  
and organize the data based on the following structure:  
```
data/dota
├── train
│   ├──images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```

# Test images
#### Split original images to 1024*1024 pixels and create COCO formate json
###### modified `DOTA_devkit/prepare_dota1.py`, mute 71-85 rows, only keep split_test
```
python DOTA_devkit/prepare_dota1.py --srcpath <dota_set_path> --dstpath <dota_1024_path>
```
where `<dota_set_path>/test/images` is required  

#### Download pre-trained weights
Download [weights](https://drive.google.com/drive/folders/1IsVLm7Yrwo18jcx0XjnCzFQQaf1WQEv8) and put the .pth file to `<dota_model_path>`  
Configs are in `configs/DOTA`

#### Test
##### Faster R-CNN
set the path of images and annotation file in `tools/test.py`  
add two line codes in row 156
```
cfg.data.test.ann_file = '<annotation_file generated from preparation step, it's a json file>'
cfg.data.test.img_prefix = '<img_path>'
```
```
python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py \
    <faster_rcnn_RoITrans_r50_fpn_1x_dota_model_path>/epoch_12.pth \ 
    --out <result_path>/results.pkl
```

##### mask R-CNN
```
python tools/test.py configs/DOTA/mask_rcnn_r50_fpn_1x_dota.py \
    <model_path>/mask_rcnn_r50_fpn_1x_dota/epoch_12.pth \
    --out <result_path>/results.pkl 
```

#### Visualize the result
##### Faster R-CNN
[Code](https://github.com/ReehcQ/satellite/blob/master/code/test.py)



# Reference 
[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
