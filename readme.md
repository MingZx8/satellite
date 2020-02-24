# Log
### _02.17 - 02.23_
+ instructions: 
  - color filter: [Matlab threshold](https://www.mathworks.com/matlabcentral/fileexchange/25682-color-threshold)
    - road
    - vehicle
+ cnn:
  - [DOTA](https://captain-whu.github.io/DOTA/dataset.html)
    - pytorch
      + installation
        - conda environment: dota3

### _02.10 - 02.16_  
+ road detection
  - [RoadNet](https://github.com/yhlleo/RoadNet)
  - [deepSeg](https://github.com/yhlleo/DeepSegmentor)
  - [U-Net](https://github.com/ArkaJU/U-Net-Satellite)
  - [airs?](https://github.com/mahmoudmohsen213/airs)
  - [road_seg](https://github.com/dariopavllo/road-segmentation)
  - [cnn](http://cs230.stanford.edu/projects_winter_2019/reports/15812659.pdf)
+ Arman's [instructions](https://github.com/ReehcQ/satellite/blob/master/instructions.md)
  - [AADT example](https://github.com/ReehcQ/satellite/blob/master/AADT%20example.csv)
  - find the date in google earth according to the locations
  - estimate the time according to the sun position: 
~~[SunEarthTool](https://www.sunearthtools.com/dp/tools/pos_sun.php)~~  
google earth sun simulator
+ google maps static api to download images
  - [pricing](https://developers.google.com/maps/documentation/maps-static/usage-and-billing) 每月免额$200？？？
  - key: AIzaSyC1tQUXzpoV0Wuj7E5ukEDfnLDDibbzxrg
  - location ex: 43.655305, -79.348989
  - zoom: 17, 18, 19, 20
    - __TODO__: test which zoom size is better for detection
    - __TODO__: convert latitude and longitude to pixel
    - __TODO__: check the captured date on Google Earth
  - scale: 2, w*h: 640 * 640, get 1280 * 1280 pixels images
+ vehicle detection
  - __TODO__: color filter
  - YOLOv3: 只有zoom20的时候可以识别出一辆车
  - __TODO__: split dataset COWC
  - __TODO__: YOLT


### _02.02 - 02.09_  
+ ~~searching sources: annotated dataset, satellite images, models...~~ 
  - [object detection sources lists](https://github.com/hoya012/deep_learning_object_detection)
  - [object detection in aerial images papers](https://github.com/murari023/awesome-aerial-object-detection#awesome-resources)
  - [satellite images related](https://github.com/robmarkcole/satellite-image-deep-learning)
  
+ ~~build [environment](https://github.com/ReehcQ/satellite/blob/master/preparation.md) for gpu-based detection models~~
  - ~~drivers~~
  - ~~tensorflow & CUDA~~
  - ~~docker~~
  
+ __Vehicle__ and __road boarder__ detection in some satellite samples
  - color filter
    + [based on sklearn](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)
  - CNN (available models)
    + [SIMRDWN(YOLT)](https://github.com/avanetten/simrdwn)
      - [ref1](https://cloud.tencent.com/developer/news/237042)
      - [ref2](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571)
      - [ref3](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
    + [SlimYOLO](https://github.com/PengyiZhang/SlimYOLOv3)
    + [Fast-RCNN based net](https://github.com/vyzboy92/Object-Detection-Net)
    + [Yolov3, here is the comparison of yolo and fast-rcnn](https://github.com/aniskoubaa/car_detection_yolo_faster_rcnn_uvsc2019)
    + [DRFBNet3000](https://github.com/seongkyun/DRFBNet300)
      - [ref](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6767679/)
    + [ResNet50](https://github.com/buzkent86/Aerial_Vehicle_Classification)
    + [RetinaNet](https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203)
      - [keras RetinaNet](https://github.com/fizyr/keras-retinanet)
      - [pedestrain detection](https://towardsdatascience.com/pedestrian-detection-in-aerial-images-using-retinanet-9053e8a72c6)
    + [U-Net](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
    
+ land-use detection
  - [1](https://github.com/datasciencecampus/laika)
  - [Mask-RCNN](https://github.com/jremillard/images-to-osm)
  - [building](https://github.com/neptune-ai/open-solution-mapping-challenge)
  - [robosat](https://github.com/mapbox/robosat) semantic segmentation


## TODO
### _02.02 - 02.09_  
1. choose a dataset where images have the similar quality to google earth/maps'
2. train and test on available models
3. compare and choose a good match
4. extract road on images


## Q   
1. vehicle classification?  
2. ~~time? some specific time for the same location (for one method: use vehicle number to estimate AADT)~~ 
1. image quality (tree branches, vehicle shadow...)?  
3. ~~other info (land use, trees, footprint...)? yes~~  what kind of land use?  
4. ~~parking cars? manually remove? if the vehicle is less than 1m to the roadside then it is parked~~  
5. area? (squares? n of images? image size?)  
6. AADT focus on highways? how about the congested roads?

# Check the project
[:)](https://github.com/ReehcQ/satellite/blob/master/intro.md)

# Reference
## models
1. [SIMRDWN](https://github.com/avanetten/simrdwn) Satellite Imagery Multiscale Rapid Detection with Windowed Networks   
__Object categories:__  
airplanes, boats, cars, and airports  
__Data source:__  
DigitalGlobe satellites, Planet satellites, and aerial platforms  
__Area:__  
?  
__Image quality:__  
0.3m/pixel (convolved from 0.15m/pixel)  
__N of objects:__  
 13,303 cars  
__mAP:__  
0.68  
__Inference Rate:__  
0.44  

## annotated dataset
1. [Awesome satellite imagery datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)  

2. [DOTA](https://captain-whu.github.io/DOAI2019/dataset.html)  
__Object categories:__   
small vehicle, large vehicle,  
plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, helicopter, roundabout, soccer ball field, swimming pool and container crane  
__Data source:__  
Google Earth, satellite JL-1, and satellite GF-2  
__Area:__  
?  
__Image quality:__  
?  
__N of objects:__  
0.4 million  
  
3. [COWC](https://gdo152.llnl.gov/cowc/)  
__Object categories:__  
car  
__Data source:__  
?  
__Area:__  
Toronto, Selwyn, Potsdam, Vaihingen, Columbus, Utah    
__Image quality:__  
0.15m per pixel    
__N of objects:__  
32,716 (58,247 negative examples)  

4. [xView](http://xviewdataset.org/)  
__Object categories:__  
60 classes  
__Data source:__  
?  
__Area:__  
?   
__Image quality:__  
0.3m per pixel    
__N of objects:__  
1 million  

5. [SpaceNet](https://spacenet.ai/)  
building detection  

6. [DroneDeploy Segmentation Dataset](https://github.com/dronedeploy/dd-ml-segmentation-benchmark)
__Object categories:__  
car, building, clutter, vegetation, water, ground  
__Data source:__  
?  
__Area:__  
?   
__Image quality:__  
0.1m per pixel    
__N of objects:__  
?   
  
## data
### data requirements

### sources  
1. [Planet](https://www.planet.com/products/planet-imagery/)  
__Resolution:__  
3m, 5m, 0.8m  
__Price:__  
?  
3-5m free trail  

2. [DigitalGlobe](https://www.digitalglobe.com/usecases#identify)

