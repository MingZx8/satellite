# Overview

# Get Started

# Image Preparation
This project uses [Google Maps API](https://cloud.google.com/maps-platform/) to download high-resolution satellite images.  
First of all, you need an **API key** to get the authentation, here is the [instruction](https://developers.google.com/maps/gmp-get-started).  
  
There are four main features regarding image size and resolution: **width**, **height**, **zoom**, **scale**.  
**zoom = 2** and **scale = 19** is an optimal option to detect vehicle in the images.  

Details about the API parameters are availble on the [developer guide](https://developers.google.com/maps/documentation/maps-static/start) page.  
    
The largest size of image that the API provided is 640*640, so this function gives a method to concatenate small size images.

![1024*1024*19*2](https://github.com/ReehcQ/satellite/blob/master/imgs/image.png)  
  
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

# Road Segmentation
The approach to detect road is purposed by [Xia et al., 2017](https://ieeexplore.ieee.org/document/8127098).   
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

#### Step 3. Filter line segments

#### Step 4. Find out pairs

#### Step 5. Cluster

#### Step 6. Build road mask and recognize route direction

# Reference
