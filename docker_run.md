```
# 도커 빌드
export AIMET_VARIANT=torch-gpu
export WORKSPACE=$(pwd)
export docker_image_name=donghn/aimet_torch:cuda12
export docker_container_name=sungmin_aimet_cuda12

docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .

# 도커 런
DATADIR=/data/dataset/

docker run -it \
  --name ${docker_container_name} \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v ${WORKSPACE}:${WORKSPACE} \
  -v ${HOME}:${HOME} \
  -w ${WORKSPACE} \
  -v "/local/mnt/workspace":"/local/mnt/workspace" \
  -v $DATADIR:"/data/dataset/" \
  --gpus all \
  --shm-size=128g \
  --ipc=host \
  --entrypoint /bin/bash \
  --hostname ${docker_container_name} \
  ${docker_image_name}




# 도커안에서 빌드
export WORKSPACE=$(pwd)
source $WORKSPACE/aimet/packaging/envsetup.sh

cd $WORKSPACE/aimet
mkdir build && cd build

cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_TENSORFLOW=OFF -DENABLE_ONNX=ON
make -j8

cd $WORKSPACE/aimet/build
make install

# 실행시마다
export WORKSPACE=$(pwd)
export PYTHONPATH=$WORKSPACE/aimet/build/staging/universal/lib/python:$PYTHONPATH


# or whl 만들어서 pip 으로 설치하기
export WORKSPACE=$(pwd)
cd $WORKSPACE/aimet/build
make packageaimet
cd packaging/dist/
pip install "PACKAGE" --no-deps 

# 일부 패키지 재설치
pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ --force-reinstall
pip install numpy==1.26.4
```