# PLAID (Protein LAtent Induced Diffusion)

## Contents

## Demo

We will provide the model on HuggingFace spaces shortly.


## Install
```
mamba env create --file environment.yaml
mamba activate plaid
```

OpenFold should be installed individually to trigger CUDA builds without downloading other dependencies:

```
git clone https://github.com/openfold/openfold.git
cd openfold
python setup.py develop
```

To develop:

```
cd plaid  # directory into which this repo is cloned
pip install -e .
```

PLAID is a latent diffusion model, whereby the ESMFold latent space is further compressed by an autoencoder. This autoencoder is offered in the [CHEAP](https://github.com/amyxlu/cheap-proteins) package. Running `pip install -r requirements.txt` using the instructions given should automatically download the CHEAP package. By default, the CHEAP autoencoder weights are saved to `~/.cache/cheap`, but this can be overriden by setting `CHEAP_CACHE=/path/to/cache`.


## Inference (Design Only)

The steps involved in PLAID inference are:
1. Sampling a latent array with length $N \times L \times 32$, where $L$ is the length of your protein, and $N$ is the number of proteins you'd like to generate.
2. Uncompress this latent array to size $N \times L \times 1024$ using the [CHEAP](https://github.com/amyxlu/cheap-proteins/tree/main) model (specifically,  `CHEAP_pfam_shorten_2_dim_32`)
3. Use the CHEAP sequence decoder to obtain the sequence.
4. Use the ESMFold structure encoder to obtain the structure.

We use Hydra to flexibly interchange between different during development. Configs can be found in `configs`, and inference configs are found in `configs/pipeline/experiments`.

To note is that for all evaluations in the paper and pipeline code, we use PLAID-2B, where the model length must be a multiple of 8. This is because the accelerated xFormers implementation only allows lengths that are multiple of 4, and we additionally use an autoencoder that reduces the length by a factor of 2. If this is an issue, you can use the PLAID-100M model.


### Mapping Conditioning Indices to GO Terms and Organisms

Conditioning is done by specifying the function and organism indices.


>![NOTE]
>To do set function to unconditional generation mode, use `2219` as the function index.
>To set organism to unconditional generation mode, use `3617` as the organism index.

You can also import the "unconditional" index using:

```
from plaid.datasets import NUM_FUNCTION_CLASSES   # 2219
from plaid.datasets import NUM_ORGANISM_CLASSES	  # 3617
```



## Inference (Evaluation)

This section lists more detailed usage instructions for those looking to reproduce the reported results. The pipeline entry points can be found in `pipeline/`. If you'd like to modify the usage pipeline, the logic is defined in `src/plaid/pipeline`.

The full pipeline (`pipeline/run_pipeline.py`) calls the model passes required to sample co-generated proteins and run designability evals:
1. Sample (`pipeline/run_sample.py`)
2. Decode (`pipeline/run_decode.py`): samples a latent, uncompresses the latent, and runs it through the sequence and structure decoders.
3. Runs ESMFold (`pipeline/run_fold.py`) on the generated sequence to obtain the "inverse generated" structure, for calculating ccRMSD.
4. Runs ProteinMPNN (`pipeline/run_inverse_fold.py`) on the generated structure to obtain the "inverse generated" sequence, for calculating ccSR.
5. Runs ProteinMPNN (`pipeline/run_inverse_fold.py`) on the "inverse generated" structure from Step 3 to obtain a "phantom generated" sequence (for calculating scSR).
6. Runs OmegaFold on the "phantom generated" sequence to obtain the structure, to obtain scRMSD (this is done simply with `omegafold <file> <outdir>`, assuming you have OmegaFold installed).

Steps 3 to 6 can be run together using `pipeline/run_consistency.py`, which is what we do for the baseline samples reported in the paper. To obtain metrics for analysis (e.g. ccRMSD, etc.), use `pipeline/run_analysis.py`.

We also provide scripts for sweeping through sampling hyperparmeters using SLURM. Scripts for running `pipeline/run_{step}.py` can be found at `scripts/eval/run_{step}.sh`.

#### Folder Structure
Using the scripts here, the output directory will have the structure:

```
root_dir
|_ generated
		|_ structures
				|_ sample1.pdb
				|_ sample2.pdb
                |_ ...
		|_ sequences.fasta
|_ inverse_generated
		|_ structures
				|_ sample1.pdb
				|_ sample2.pdb
                |_ ...
		|_ sequences.fasta
|_ phantom_generated
		|_ structures
				|_ sample1.pdb
				|_ sample2.pdb
                |_ ...
		|_ sequences.fasta
|_ designability.csv
```

`designability.csv` is not generated automatically by the pipeline; `run_analysis.py` needs to be run separately.

#### Hydra Configuration

Sample configs for common usage patterns can be found in `config/pipeline/experiment`.

For better control of the pipeline (e.g. batch sizes, length, sampling hyperparameters), see the [Hydra docs](https://hydra.cc/docs/intro/). Briefly, the 'main' config for the inference pipeline is specified in `config/pipeline/consistency.yaml`, with 


## Training
We make our training code available to encourage others to use the PLAID paradigm of multimodal generation via latent space diffusion with their predition models. DDP uses the PyTorch Lightning package; this code is tested for up to 10 nodes of 8 A100s. To launch with the default settings used to train our primary model:

```
```

The code also includes support for:

* Min-SNR loss scaling
* Classifier-free guidance (with GO term and organism)
* Self-Conditioning

If using `torch.compile`, please make sure to use `float32` rather than mix precision or `bfloat16` due to [this issue](https://github.com/facebookresearch/xformers/issues/920) in the `xFormers` library.


### Data availability
Data used to train the model can be found on HuggingFace, including:
* Training dataset as TAR-compressed shards, for use with WebDataset
* validation data parquet file (for running Sinkhorn distance evaluations)
