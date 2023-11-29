# k-diffusion

An implementation of [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) for PyTorch, with enhancements and additional features, such as improved sampling algorithms and transformer-based diffusion models.

## Useful Commands
`find . -type f -mtime +XXX -exec rm {} \;`
`find . -type f -mtime +4`


To run an image that mounts locally (on DGX5) as root user:
```
docker run -it --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion mambaorg/micromamba:jammy-cuda-12.2.0 /bin/bash

# if nvidia-container-toolkit is installed
docker run -it --name mambacuda --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion mambaorg/micromamba:jammy-cuda-12.2.0 /bin/bash
```
Replace `source="..."` with the source directory on your local machine.

This base image will use `micromamba` instead of `conda`, but most common conda commands can be used by directly swapping out `conda` for `micromamba`.

```
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

then running:

```sh
$ accelerate launch train.py --config CONFIG_FILE --name RUN_NAME
```

## Enhancements/additional features (h/t Katherine Crowson)

- k-diffusion has support for training transformer-based diffusion models (like [DiT](https://arxiv.org/abs/2212.09748) but improved).

- k-diffusion supports a soft version of [Min-SNR loss weighting](https://arxiv.org/abs/2303.09556) for improved training at high resolutions with less hyperparameters than the loss weighting used in Karras et al. (2022).

- k-diffusion has wrappers for [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch), [OpenAI diffusion](https://github.com/openai/guided-diffusion), and [CompVis diffusion](https://github.com/CompVis/latent-diffusion) models allowing them to be used with its samplers and ODE/SDE.

- k-diffusion implements [DPM-Solver](https://arxiv.org/abs/2206.00927), which produces higher quality samples at the same number of function evalutions as Karras Algorithm 2, as well as supporting adaptive step size control. [DPM-Solver++(2S) and (2M)](https://arxiv.org/abs/2211.01095) are implemented now too for improved quality with low numbers of steps.

- k-diffusion supports [CLIP](https://openai.com/blog/clip/) guided sampling from unconditional diffusion models (see `sample_clip_guided.py`).

- k-diffusion supports log likelihood calculation (not a variational lower bound) for native models and all wrapped models.

- k-diffusion can calculate, during training, the [FID](https://papers.nips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf) and [KID](https://arxiv.org/abs/1801.01401) vs the training set.

- k-diffusion can calculate, during training, the gradient noise scale (1 / SNR), from _An Empirical Model of Large-Batch Training_, https://arxiv.org/abs/1812.06162).


