1. ~~Check if you can extract time and date from google api images~~
~~(please just check the website and/or try to extract it only for one point).~~  
__google maps does not provide captured date__

2. ~~If this is possible to extract the date and hour from google api,~~ 
~~then please send me all times and dates of google api images for the coordinates I sent to you. 
I will check city database to see if there are observed hourly traffic data (C(obs)) for the dates
you have found google api images~~

3. ~~If we can not extract time and date of image from google api.~~
~~Please check google earth image. 
Please check if we can extract date and time for google earth images 
(please just check the website and/or try to extract it only for one point).~~  
__google earth has the historical imagery, and google maps show the image with the best quality__  
__for some reason, it does not show the vehicle on the road__  
__Can we choose the image from google earth?__  

4. ~~If this is possible to extract the date and hour from google earth,~~ 
~~then please send me all times and dates of google earth images for the coordinates I sent to you. 
I will check city database to see if there are observed hourly traffic data (C(obs)) for the dates 
you have found google earth images.~~

__Check the sun position and estimate the time according to the shadow:__  
__[SunEarthTool](https://www.sunearthtools.com/dp/tools/pos_sun.php)__

5. If I found enough data (from steps 2 / 3) then I will ask you to count vehicles (C(x)) on the closet road 
to the coordinates using image processing. We also need to find the road length (x) at the same time from these images.

6. Then, I will send you a shape-file that contains all average vehicle speed on roads.

7. From the road length (x) and vehicle speed (v) 
we can approximate the time (t) that all vehicle passed the station (coordinate is the location of station). 
t=x/v  à assume that t is in minute

8. Then, we approximate the hourly traffic count (C(h)) using C(h)=(C(x)/t) * 60

9. Then, we compared observed hourly traffic counts (C(obs) from step 2 or 4) with C(h) (step 8).

10. Finally, and if approximated hourly values (C(h)) are good, we use TEPs to find AADTs.
