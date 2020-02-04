# Log
_02.02 - 02.09_  
1. searching sources: annotated dataset, satellite images, models...  
  
__Q:__    
1. vehicle classification?  
2. time? image quality (tree branches, vehicle shadow...)?  
3. other info (land use, trees, footprint...)? 
4. parking cars? manually remove?  
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




## data
### data requirements

### sources  
1. [Planet](https://www.planet.com/products/planet-imagery/)  
__Resolution:__  
3m, 5m, 0.8m  
__Price:__  
?  
3-5m free trail  
