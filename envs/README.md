Quick notes for setting up docker:
* using the micromamba base image:
```
docker run -it --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion mambaorg/micromamba:jammy-cuda-12.2.0 /bin/bash
docker run -it --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion anibali/pytorch:2.0.1-cuda11.8 /bin/bash
docker run -it --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime /bin/bash
docker run -it --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel /bin/bash

```


within the container:
```
micromamba install -c conda-forge gcc tmux --yes
micromamba install -c anaconda git wget --yes
```

install OpenFold (ish) -- 

need to install GCC before some pip packages:

`micromamba install -c conda-forge tmux gcc --yes`

For some reason, nvcc doesn't show up
micromamba install -c conda-forge cudatoolkit=11.8