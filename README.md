## Introduction
tensorrt-inference-server 部署onnx以及torchscript模型

## Prerequisites
安装有nvidia-docker 的Ubuntu系统

## Docker
- `docker pull yingchao126/tensorrtserver:20.02-py3`
- `docker pull yingchao126/tensorrtserver:20.02-py3-clientsdk`

## Usage

```
运行服务器
docker run -it --gpus all --shm-size=4g --ulimit memlock=-1 --name trtserver --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /{your-path}/model_repository/:/models yingchao126/tensorrtserver:20.02-py3 trtserver --model-repository=/models
运行客户端
docker run -it --gpus all --rm --net=host yingchao126/tensorrtserver:20.02-py3-clientsdk
可以把里边的python包提取出来，就可以在外边使用了/workspace/install/python tensorrtserver-1.11.0-py3-none-linux_x86_64.whl
pip install tensorrtserver-1.11.0-py3-none-linux_x86_64.whl
之后就可以运行客户端测试demo
python video_client.py

```
