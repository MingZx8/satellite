1. run docker under non-root user [(ref)](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user)  
``` sudo groupadd docker```  
``` sudo usermod -aG docker $USER```  
``` newgrp docker```  
``` docker run hello-world```  

2. list containers and remove it  
``` docker container ls --all```  
``` docker container rm <containerID>```  

