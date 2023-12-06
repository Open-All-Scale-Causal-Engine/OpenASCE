# Installation Guide

The most part of OpenASCE is written in Python, and the installation process of this part is quite simple. While the causal tree algorithms rely on C++ libraries, currently we only provide pre-built binary wheels for Linux(x86_64). We also provide a Docker image which includes all the necessary libraries for users to build and install OpenASCE from the source.

## Pip Installation

### Linux

For Linux (x86_64) with Python 3.11, the pre-built wheel is uploaded to PyPI (Python Package Index), users could simply install it as follows.

```bash
pip install openasce
```

We will support more platforms later.

### Other Platforms

For other platforms, users can install from the source code as follows.

```bash
git clone https://github.com/Open-All-Scale-Causal-Engine/OpenASCE.git
cd OpenASCE
pip install .
```

After the installation, you can use all the algorithms except causal tree algorithms.

## Docker Installation

We provide a docker image that contains all the dependencies required to build the C++ code and run OpenASCE. Users can practice the system using the image without environmental issues [^1].

[^1]: Currently it doesn't work well on Mac with a Silicon chip.

Before getting the image, the user should log in to the Aliyun Registry first. For users without an Aliyun account, please refer to [this link](https://account.aliyun.com/) for registration.

```bash
docker login --username=${your_user_name} registry.cn-hangzhou.aliyuncs.com
docker pull registry.cn-hangzhou.aliyuncs.com/openasce/openasce:gcc9.4-py3.11
```

You may need to use `sudo` if you are using a non-root user.

The users can also build their image for OpenASCE according to the [Dockerfile](https://github.com/Open-All-Scale-Causal-Engine/OpenASCE/blob/main/docker/Dockerfile).

```bash
docker build --network host -f Dockerfile -t openasce:gcc9.4-py3.11 .
```

### 1. Start Docker

After getting the image, the user can run the container as follows.

```bash
# pull from the registry
docker run --net=host --rm -it -m 16g --name openasce_env registry.cn-hangzhou.aliyuncs.com/openasce/openasce:gcc9.4-py3.11 "/usr/bin/bash"
# build from the local
docker run --net=host --rm -it -m 16g --name openasce_env openasce:gcc9.4-py3.11 "/usr/bin/bash"
```

### 2. Clone the Code

```bash
git clone https://github.com/Open-All-Scale-Causal-Engine/OpenASCE.git
cd OpenASCE
git submodule update --recursive --init
```

### 3. Compile the Source Code and Install the Package

You can use the `build.sh` script to build the C++ code and install OpenASCE in the current environment.

```
bash scripts/build.sh dev
```

The package can be found in the dist directory and is typically named openasce-0.1.0-cp311-none-linux_x86_64.whl for reinstallation.
