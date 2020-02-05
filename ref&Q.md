### Articles
__Estimating Annual Average Daily Traffic from Satellite Imagery and Air Photos__  
+ AADT represents traffic on a __highway__ segment on an average day  
+ difficlut to obtain coz it is based on dynamic traffic conditions  
+ data source and quality: IKONOS __1-m__ resolution panchromatic image  
+ One issue is the error involved when a single snapshot is used to infer average traffic conditions (converting images of traffic highway segments to AADT
estimates and comparing these estimates with those that are available from traditional means of AADT estimation)    
+ The other issue relates to whether the errors involved with single snapshots can be reduced sufficiently to be of use in AADT
estimation when combining data from several images of the same highway segment  

_a single AADT estimate is produced from the image_

/ | Steps
--- | --- 
1 | Obtain the __vehicle density__ from the image
2 | Convert the density to a __short-duration (t-minute) volume__
3 | Expand the t-minute volume to an __hourly volume__
4 | Expand the hourly volume to a __daily volume__
5 | Deseasonalize the daily volume to produce an __average yearly volume__
