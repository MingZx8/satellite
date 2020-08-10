# Overview

# Get Started

# Image Preparation
This project uses [Google Maps API](https://cloud.google.com/maps-platform/) to download high-resolution satellite images.  
First of all, you need an **API key** to get the authentation, here is the [instruction](https://developers.google.com/maps/gmp-get-started).  
  
There are four main features regarding image size and resolution: **width**, **height**, **zoom**, **scale**.  
**zoom = 2** and **scale = 19** is an optimal option to detect vehicle in the images.  

Details about the API parameters are availble on the [developer guide](https://developers.google.com/maps/documentation/maps-static/start) page.  
    
The largest size of image that the API provided is 640*640, so this function gives a method to concatenate small size images.
  
The satellite images with high resolution is always available for main city like Toronto.  
In a few part of Quebec, only low quality images or even no image are provided by Google Maps.

## [Code](https://github.com/ReehcQ/satellite/blob/master/code/download.py)
Replace the variable `key` with your API key.  
Use the function `download` to download satellite image.  

```
download(43.6659008, -79.3928523, 2048, 2048, 2, 19, '\path', 'APIkey')
```
  
###### So far, this function does not support neither the width nor the height of the image is smaller than 640


# Vehicle Detection

# Road Segmentation
