# Overview
This is a transportation and environment project from TRAQ group in the University of Toronto.  
This project is purposed to detect and **count traffic** in the **satellite images** in order to estimate AADT for road segments. Compared to traditional manual methods of traffic collection, this approach is more efficient and economic.  
  
![result](https://github.com/MingZx8/satellite/blob/master/imgs/result.png)
  
The approach consists of three main steps: 
+ [Image preparation](https://github.com/ReehcQ/satellite#image-preparation-1)
+ [Vehicle detection](https://github.com/ReehcQ/satellite#vehicle-detection-1)
+ [Road segmentation](https://github.com/ReehcQ/satellite#road-segmentation-1)
  
This documentation covers the requirements for each step, the basic instructions to get detection from a satellite image, and the details for each step if processing in batch is needed.
  
# Prerequisite
This project has been tested on the following dependencies:
  
#### overall
+ [Python](https://www.python.org/) 3.6
+ [opencv-python](https://pypi.org/project/opencv-python/)
+ [shapely](https://pypi.org/project/Shapely/)
#### Image preparation
+ [urllib3](https://urllib3.readthedocs.io/en/latest/)
+ [Google Maps API](https://cloud.google.com/maps-platform/)
#### Vehicle detection
+ Linux ([Ubuntu](https://ubuntu.com/) 18.04)
+ [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
+ [GCC](https://gcc.gnu.org/) 7.5.0
+ [Pytorch](https://pytorch.org/) 1.3.1
+ [AerialDetection](https://github.com/dingjiansw101/AerialDetection/blob/master/INSTALL.md)
+ [NN model](https://drive.google.com/drive/folders/1VYygIQNSsXr8Ij5B9GjqsKbNMjLuMG-x)
#### Road segmentation
+ [LSD](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) (Line segment detection)
+ Geospatial data: 
  - [Ontario](https://mdl.library.utoronto.ca/collections/geospatial-data/canmap-routelogistics-ontario-5)
  - [Quebec](https://mdl.library.utoronto.ca/collections/geospatial-data/canmap-routelogistics-quebec-3) 
  - (please find '*all road files directions speed limits travel time*')
+ [scikit-learn](https://scikit-learn.org/stable/)

  

# Get Started
### Preparation
Please check first if you have:
+ Linux
+ python 3.6
+ [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
+ [GCC](https://gcc.gnu.org/) 7.5.0
+ [Pytorch](https://pytorch.org/) 1.3.1
+ [opencv-python](https://pypi.org/project/opencv-python/)
+ [urllib3](https://urllib3.readthedocs.io/en/latest/)
+ [scikit-learn](https://scikit-learn.org/stable/)
+ [shapely](https://pypi.org/project/Shapely/)
  
Use `git clone https://github.com/MingZx8/satellite.git` to save this repository or download/extract the zip file directly. The path is denoted as `<repo_path>` below.  

This project uses [Google Maps API](https://cloud.google.com/maps-platform/) to download high-resolution satellite images.  
First of all, you need an **API key** to get the authentation, here is the [instruction](https://developers.google.com/maps/gmp-get-started).  
Replace the API_KEY variable with your API key in the file `<repo_path>/code/download.py`.  

Download the geospatial data to the local path `<geo_path>`.  

Download the line segment detection [tool](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi), save and extract it to your `<LSD_path>`. In terminal, go to the folder and run command 'make'.

Clone and install the [AerialDetection](https://github.com/dingjiansw101/AerialDetection/blob/master/INSTALL.md) tool to the local, named `<AerialDetection_path>`.  
In order to customize input file path, replace the file `<AerialDetection_path>/tools/test.py` with the file `<repo_path>/code/test.py` or [this python file](https://github.com/ReehcQ/satellite/blob/master/code/test.py).  

Download the [model](https://drive.google.com/drive/folders/1VYygIQNSsXr8Ij5B9GjqsKbNMjLuMG-x) and save it to your `<model_path>`.  

In order to call the detection function, go to file `<repo_path>/detect_vehicle.py`
+ replace the variable `model` with `<model_path>`.
+ replace the variable `dota_path` with `<AerialDetection_path>`

To call the line segment function, go to file `<repo_path>/generateLSD.py`, replace the variable `EXECUTE_PATH` with `<LSD_path>/lsd`.  
  
### Single image detection
Now you can run the program with the entry function in the file `<repo_path>/code/count_vehicle.py`, e.g.:  
```
main(43.659435, 
     -79.354539,
     2048,
     1024,
     19,
     2,
     '../output',
     centreline_label='eg',
     geo_file='../ONrte/ONrte.shp')
```
The vehicle count result will show in `<output_path>/count_station.csv` and visually show as `<output_path>/count.csv`.    


### Process multiple images step by step
It will be time-consuming since the detection algorithm is initiated when every image is comming in. Processing images in batch (around 2000 images) is recommended.  

#### Image Preparation
Use the function `download` in the file `<repo_path>/code/download.py`, e.g.:  
`download(43.659435, -79.354539, 2048, 1024, 19, 2, output_path='../output', centreline_label='eg')`  
params:
```
download(  
  latitude=43.659435,   
  longitude=-79.354539,   
  width=2048,   
  height=1024,   
  scale=19,   
  zoom=2,   
  output_path='../output',   
  centreline_label='eg')
```  
The directory tree is like:  
```
- _output
  - _eg
    - _image
      - image.png
  - _eg2
    - _image
      - image.png
```  
copy the .png files out and put them into a folder named `<image_folder>/image`.  

#### Vehicle detection
Go to the file `<repo_path>/code/detect_vehile.py`.  
```
main(<image_folder>)
```
Then you have the detection result `<image_folder>/file/test.csv`.  
In this table, the characters before the first underscore of values in the column 'img' stands for the image where the vehicle shows.  
So you need to split the table for each image, and put then back to the original image folder.  
Let the directory tree look like:  
```
- _output
  - _eg
    - _file
      - test.csv
    - _image
      - image.png
  - _eg2
    - _file
      - test.csv
    - _image
      - image.png
```  

#### Road mask
Now you can use a loop with the entry function in the file `<repo_path>/code/count_vehicle.py` for each image.  
```
main(43.659435, 
     -79.354539,
     2048,
     1024,
     19,
     2,
     '../output',
     centreline_label='eg',
     geo_file='../ONrte/ONrte.shp')
```
  
# Image Preparation
There are four main features regarding image size and resolution: **width**, **height**, **zoom**, **scale**.  
**zoom = 2** and **scale = 19** is an optimal option to detect vehicle in the images.  

Details about the API parameters are availble on the [developer guide](https://developers.google.com/maps/documentation/maps-static/start) page.  
    
The largest size of image that the API provided is 640*640, so this function gives a method to concatenate small size images.

![1024\*1024\*19\*2](https://github.com/ReehcQ/satellite/blob/master/imgs/image.png)  
  
The satellite images with high resolution is always available for main city like Toronto.  
In a few part of Quebec, only low quality images or even no image are provided by Google Maps.

[Code](https://github.com/ReehcQ/satellite/blob/master/code/download.py)  
Replace the variable `key` with your API key.  
Use the function `download` to download satellite image.  

```
download(43.6659008, -79.3928523, 2048, 2048, 2, 19, <output directory>, <APIkey>, <--optional centreline_label='file name' >)
```
  
###### So far, this function does not support neither the width nor the height of the image is smaller than 640

  
# Vehicle Detection
[Code](https://github.com/ReehcQ/satellite/blob/master/code/detect_vehicle.py)  
This process includes four steps:  
### Split image
The image will be split into 1024\*1024, and it will generate a .json file including split img name and its features such as size.  
```
split_img(<file path>)
```
  
### Detect vehicles
```
detect_car(<file path>)
```
  
### Output results
The previous step gives a .pkl file. We will convert it into a DataFrame.
```
pkl2csv(<file path>)
```
  
### Merge results
Since the original image was split into several 1024\*1024 images, we need to concatenate these images together and merge the detection results. 
```
merge(<file path>, <--optional bool show_img>)
```
You can use `main(<file path>, <--optional bool show_img>)` function to run these processes in a serie.
  
  
  
# Road Segmentation
The approach to mesure road width is purposed by [Xia et al., 2017](https://ieeexplore.ieee.org/document/8127098).   
From the geospatial data, road type and road centerline are determined. A road mask is built based on the road width and centerline.  
Here are the steps:
#### Step 1. Line segment detection
![step 1](https://github.com/ReehcQ/satellite/blob/master/imgs/step1.png)
[Line Segment Detector](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) is used to detect line segment (straight contour) in the image.  
Download the tool. In terminal, go to the folder and run command 'make'.

[Code](https://github.com/ReehcQ/satellite/blob/master/code/generateLSD.py)  
Download the above-mentioned [tool](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) and save it to your *\<LSD_path>*.
```
generate(<file path>, <LSD_path>)
```
  
#### Step 2. Select area according to the Geospatial data
![step 2](https://github.com/ReehcQ/satellite/blob/master/imgs/step2.png)  
In this project, geographical information is used to determine the road centerline.  
The default width is given according to the type of road.  
Avoiding overlapping other road is considered. 
Only line segments in this area will be paired.  
  
#### Step 3. Filter line segments
![step 3](https://github.com/ReehcQ/satellite/blob/master/imgs/step3.png)  
We filter out line segment out of the selected area and its length should be larger than a selected length.  
  
#### Step 4. Find out pairs
![step 4](https://github.com/ReehcQ/satellite/blob/master/imgs/step4.png)  
We calculate overlap ratio for all segment pairs and keep that whose overlap ratio is larger than 0.2.  
*Overlap ratio:*
+ *sample target segment at fixed step, the number of sample points is m*
+ *draw lines perpendicular to candidate line (where candidate segment located in), over sample point on the target segment*
+ *count the perpendicular lines across the candidate segment, n*
+ *overlap ratio is n/m*  
![step 4-2](https://github.com/ReehcQ/satellite/blob/master/imgs/step4-2.png)  
  
#### Step 5. Cluster
![step 5](https://github.com/ReehcQ/satellite/blob/master/imgs/step5.png)  
We can get a road width estimation using k-mean clustering method.  
  
#### Step 6. Build road mask and recognize route direction
![step 6](https://github.com/ReehcQ/satellite/blob/master/imgs/step6.png) 
  
[Code](https://github.com/ReehcQ/satellite/blob/master/code/road_mask.py)  

# Reference
## [DOTA (A Large-scale Dataset for Object DeTection in Aerial Images)](https://captain-whu.github.io/DOTA/)
```
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}

@InProceedings{Ding_2019_CVPR,
author = {Ding, Jian and Xue, Nan and Long, Yang and Xia, Gui-Song and Lu, Qikai},
title = {Learning RoI Transformer for Oriented Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
## [LSD (Line Segment Detection)](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi)
```
Grompone von Gioi, R., Jakubowicz, J., Morel, J.-M., & Randall, G. (2012). LSD: a Line Segment Detector. Image Processing On Line, 2, 35–55. https://doi.org/10.5201/ipol.2012.gjmr-lsd
```
## [Road width measurement from remote sensing images](https://uwaterloo.ca/geospatial-sensing/sites/ca.geospatial-sensing/files/uploads/files/0000902.pdf)
```
Z. Xia, Y. Zang, C. Wang and J. Li, "Road width measurement from remote sensing images," 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), Fort Worth, TX, 2017, pp. 902-905, doi: 10.1109/IGARSS.2017.8127098.
```
