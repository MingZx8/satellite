# Overview
This project is purposed to detect and **count traffic** in the **satellite images** in order to estimate AADT for road segments. Compared to traditional manual methods of traffic collection, this approach is more efficient and economic.  
  
The approach consists of three main steps: 
+ [Image preparation](https://github.com/ReehcQ/satellite#image-preparation-1)
+ [Vehicle detection](https://github.com/ReehcQ/satellite#vehicle-detection-1)
+ [Road segmentation](https://github.com/ReehcQ/satellite#road-segmentation-1)
  
This documentation covers the requirements for each step, the basic instructions to get detection from a satellite image, and the details of each step if processing in batch is needed.
  
# Prerequisite
This project has been tested on the following dependencies:
  
#### overall
+ [Python](https://www.python.org/) 3.6
+ [opencv-python](https://pypi.org/project/opencv-python/)
#### Image preparation
+ [urllib3](https://urllib3.readthedocs.io/en/latest/)
#### Vehicle detection
+ Linux ([Ubuntu](https://ubuntu.com/) 18.04)
+ [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
+ [GCC](https://gcc.gnu.org/) 7.5.0
+ [Pytorch](https://pytorch.org/) 1.3.1
+ [AerialDetection](https://github.com/dingjiansw101/AerialDetection/blob/master/INSTALL.md)
+ [NN model](https://drive.google.com/drive/folders/1VYygIQNSsXr8Ij5B9GjqsKbNMjLuMG-x)
+ [shapely](https://pypi.org/project/Shapely/)
#### Road segmentation
+ [LSD](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) (Line segment detection)
+ Geospatial data (such as shapefile)
+ [shapely](https://pypi.org/project/Shapely/)
+ [scikit-learn](https://scikit-learn.org/stable/)

  

# Get Started

  
# Image Preparation
This project uses [Google Maps API](https://cloud.google.com/maps-platform/) to download high-resolution satellite images.  
First of all, you need an **API key** to get the authentation, here is the [instruction](https://developers.google.com/maps/gmp-get-started).  
  
There are four main features regarding image size and resolution: **width**, **height**, **zoom**, **scale**.  
**zoom = 2** and **scale = 19** is an optimal option to detect vehicle in the images.  

Details about the API parameters are availble on the [developer guide](https://developers.google.com/maps/documentation/maps-static/start) page.  
    
The largest size of image that the API provided is 640*640, so this function gives a method to concatenate small size images.

![1024\*1024\*19\*2](https://github.com/ReehcQ/satellite/blob/master/imgs/image.png)  
  
The satellite images with high resolution is always available for main city like Toronto.  
In a few part of Quebec, only low quality images or even no image are provided by Google Maps.

## [Code](https://github.com/ReehcQ/satellite/blob/master/code/download.py)
Replace the variable `key` with your API key.  
Use the function `download` to download satellite image.  

```
download(43.6659008, -79.3928523, 2048, 2048, 2, 19, <output path>, <APIkey>)
```
  
###### So far, this function does not support neither the width nor the height of the image is smaller than 640

  
# Vehicle Detection
## [Code](https://github.com/ReehcQ/satellite/blob/master/code/detect_vehicle.py)
This process includes four steps:  
### Split image
The image will be split into 1024\*1024, and it will generate a .json file including split img name and its features such as size.  
```
split_img()
```
  
### Detect vehicles
In order to customize input file path, replace the file <../AerialDetection/tools/test.py> with [this python file](https://github.com/ReehcQ/satellite/blob/master/code/test.py).  
```
detect_car()
```
  
### Output results
The previous step gives a .pkl file. We will convert it into a DataFrame.
```
pkl2csv()
```
  
### Merge results
Since the original image was split into several 1024\*1024 images, we need to concatenate these images together and merge the detection results. 
```
merge()
```
You can use main() function to run these processes in a serie.
  
# Road Segmentation
The approach to mesure road width is purposed by [Xia et al., 2017](https://ieeexplore.ieee.org/document/8127098).   
From the geospatial data, road type and road centerline are determined. A road mask is built based on the road width and centerline.  
Here are the steps:
#### Step 1. Line segment detection
![step 1](https://github.com/ReehcQ/satellite/blob/master/imgs/step1.png)
[Line Segment Detector](http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) is used to detect line segment (straight contour) in the image.    

## [Code](https://github.com/ReehcQ/satellite/blob/master/code/generateLSD.py)
Download the code and save it to your *\<LSD path>*.
```
generate(<img.jpg>, <output_folder_path>, <LSD path>)
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
  
## [Code](https://github.com/ReehcQ/satellite/blob/master/code/road_mask.py)


# Reference
