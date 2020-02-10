# prerequisites
System: Ubuntu 18.04  
GPU:  
1. GTX 745
2. RTX 2070  

Python 3.5

## NVIDIA and N-GPU
#### *uninstall drivers
`sudo apt-get purge nvidia-*`  
`sudo apt autoremove`  

#### mute integrated graphics
`cd /etc/modprobe.d`  
`sudo nano blacklist.conf`  
add `blacklist nouveau`  
add `options nouveau modeset=0`  

#### install graphic drivers
`sudo add-apt-repository ppa:graphics-drivers/ppa`  
`sudo apt-get update`  
`sudo apt-get install nvidia-driver-440`  

##### problems may occur 
###### lightdm & lightdm.service problem: reinstall lightdm and startx
(choose lightdm rather than other display managers)   
`sudo apt-get purge lightdm`  
`sudo apt-get install lightdm`  
`sudo service lightdm restart`  
  
###### X11/xorg
`sudo apt-get install xserver-xorg-core xorg openbox`  
  
###### lost display
`sudo apt-get install ubuntu-desktop`  


## docker and NVIDIA-docker
recommand nvidia-docker2  
`sudo apt-get install nvidia-docker2`  
`sudo pkill -SIGHUP dockerd`
###### remove old nvidia-docker
`docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f`  
`sudo apt-get purge nvidia-docker`

## tensorflow-gpu and CUDA
#### install tensorflow-gpu
`pip install tensorflow-gpu`

#### install CUDA
[Download CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)  
Follow the instruction to install

#### *remove and replace the old CUDA

