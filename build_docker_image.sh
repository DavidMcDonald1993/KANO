#!/bin/bash 

IMAGE_NAME=davidmcdonald93/docker-remote:kano
BUILD_IMAGE_NAME=${IMAGE_NAME}-build

# build and tag build image
sudo docker build --target build -t ${BUILD_IMAGE_NAME} .
# build and tag runtime image
sudo docker build -t ${IMAGE_NAME} .
# push runtime image
sudo docker push ${IMAGE_NAME}

# remove all exited containers
sudo docker rm $(sudo docker ps --filter=status=exited --filter=status=created -q)

# remove all untagged images 
sudo docker rmi $(sudo docker images -a --filter=dangling=true -q)