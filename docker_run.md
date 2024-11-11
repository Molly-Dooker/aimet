```
export WORKSPACE=$(pwd)
export AIMET_VARIANT=torch-gpu
export docker_image_name=aimet_cuda12:v1.0
export docker_container_name=sungmin_aimet_cuda12

# docker build
docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .

# docker run
docker run -it --gpus all --name ${docker_container_name} -u $(id -u ${USER}):$(id -g ${USER}) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
  -v "/local/mnt/workspace":"/local/mnt/workspace" \
  --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}

  
```