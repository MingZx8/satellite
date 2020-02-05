# Log
_02.02 - 02.09_  
+ ~~searching sources: annotated dataset, satellite images, models...~~   
+ vehicle detection in some sample satellite images
  
__Q:__    
1. vehicle classification?  
2. ~~time? some specific time for the same location (for one method: use vehicle number to estimate AADT)~~ 
1. image quality (tree branches, vehicle shadow...)?  
3. ~~other info (land use, trees, footprint...)? yes~~  what kind of land use?  
4. ~~parking cars? manually remove? if the vehicle is less than 1m to the roadside then it is parked~~  
5. area? (squares? n of images? image size?)  


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

