# PLAID (Protein LAtent Induced Diffusion)

## Contents

## Demo

We will provide the model on HuggingFace spaces shortly.


## Installation

Clone the repository:

```
git clone https://github.com/amyxlu/plaid.git
cd plaid
```

To create the environment for the repository:
```
conda env create --file environment.yaml   # create environment
pip install --no-deps git+https://github.com/amyxlu/openfold.git  # installs openfold
pip install -e .   # installs PLAID
```

The ESMFold structure module use the OpenFold implementation, which includes custom CUDA kernels for the attention mechanism. Installing using the instructions here will automatically install an OpenFold [fork]() in no-dependency mode, which includes some minor changes to use C++17 instead of C++14 to build the CUDA kernels, for compatibility with `torch >= 2.0`.

PLAID is a latent diffusion model, whereby the ESMFold latent space is further compressed by an autoencoder. This autoencoder is offered in the [CHEAP](https://github.com/amyxlu/cheap-proteins) package. Using the instructions given should automatically download the CHEAP package.
>[!TIP]
>By default, the PLAID model weights are saved to `~/.cache/plaid`, and CHEAP autoencoder weights are saved to `~/.cache/cheap`. You can override this by setting environment variables `CHEAP_CACHE=/path/to/cache` and `PLAID_CACHE=/path/to/cache`.
>```
>echo "export CHEAP_CACHE=/data/lux70/cheap" >> ~/.bashrc
>```


## Inference (Design-Only)

This section includes details for sampling from PLAID; if you are looking for more detailed instructions in evaluating samples, see Inference (Evaluation).

The steps involved in PLAID generation are:
1. **Sample** (`pipeline/run_sample.py`): samples a latent embedding with reverse diffusion.
2. **Decode** (`pipeline/run_decode.py`): uncompresses the latent, and runs it through the sequence and structure decoders.

We use Hydra for configuration. Inference configs are found in `configs/pipeline/experiments`, and we suggest reading the documentation to better understand the usage syntax described.

To run the sample and decoding together:


### Step 1: Sampling the Latent Embedding
Sampling a latent array with length $N \times L \times 32$, where $L$ is the length of your protein, and $N$ is the number of proteins you'd like to generate.

If the GO term (i.e. `function_idx`) is specified, you can use `length=None` to automatically infer what an appropriate sequence length should be, by randomly sampling a Pfam family that uses the given GO term.

You can use this stage as a standalone script using Hydra syntax. Options are specified in `configs/pipeline/sample/sample_latent.yaml`. You can also use one of the other common use case configs in `configs/pipeline/sample/`, such as `ddim_unconditional.yaml`, which is for unconditional sampling. 

```
# conditional prompt: human AND 6-phosphofructokinase activity, automatically infer length
python pipeline/run_sample.py ++length=null ++function_idx=166 ++organism_idx=1326

# conditional prompt: human AND 6-phosphofructokinase activity, always use length 400
# NOTE: the length specified here most be automatically infer length
python pipeline/run_sample.py ++length=200 ++function_idx=166 ++organism_idx=1326

# unconditional generation; specify output directory
python pipeline/run_sample.py ++length=200 ++function_idx=2219 ++organism_idx=3617 ++output_root_dir=/data/lux70/plaid/samples/unconditional
```

>[!IMPORTANT]
>The length specified here is **half the actual length**, and **must be divisible by 4**.
>For example, if you want to sample a protein with length 200, you should pass `length=100`.

For all evaluations in the paper and pipeline code, we use PLAID-2B, where the model length must be a multiple of 8. This is because the accelerated xFormers implementation only allows lengths that are multiple of 4, and we additionally use an autoencoder that reduces the length by a factor of 2. If this is an issue, you can use the PLAID-100M model.

The main logic for the sampling is defined in in the `SampleLatent` class in `src/pipeline/_sample.py`; the class can also be imported to be used in a pipeline, as is the case in `pipeline/run_pipeline.py`.

### Mapping Conditioning Indices to GO Terms and Organisms

Conditioning is done by specifying the function and organism indices.


>[!NOTE]
>To do set function to unconditional generation mode, use `2219` as the function index.
>To set organism to unconditional generation mode, use `3617` as the organism index.

You can also import the "unconditional" index using:

```
from plaid.datasets import NUM_FUNCTION_CLASSES   # 2219
from plaid.datasets import NUM_ORGANISM_CLASSES	  # 3617
```


### Step 2-4: Obtain the Sequence and Structure


2. Uncompress this latent array to size $N \times L \times 1024$ using the [CHEAP](https://github.com/amyxlu/cheap-proteins/tree/main) model (specifically,  `CHEAP_pfam_shorten_2_dim_32`)
3. Use the CHEAP sequence decoder to obtain the sequence.
4. Use the ESMFold structure encoder to obtain the structure.

## Inference (Evaluation)

This section lists more detailed usage instructions for those looking to reproduce the reported results. The pipeline entry points can be found in `pipeline/`. If you'd like to modify the usage pipeline, the logic is defined in `src/plaid/pipeline`.

The full pipeline (`pipeline/run_pipeline.py`) calls the model passes required to sample co-generated proteins and run designability evals:
1. Sample (`pipeline/run_sample.py`)
2. Decode (`pipeline/run_decode.py`): samples a latent, uncompresses the latent, and runs it through the sequence and structure decoders.
3. Runs ESMFold (`pipeline/run_fold.py`) on the generated sequence to obtain the "inverse generated" structure, for calculating ccRMSD.
4. Runs ProteinMPNN (`pipeline/run_inverse_fold.py`) on the generated structure to obtain the "inverse generated" sequence, for calculating ccSR.
5. Runs ProteinMPNN (`pipeline/run_inverse_fold.py`) on the "inverse generated" structure from Step 3 to obtain a "phantom generated" sequence (for calculating scSR).
6. Runs OmegaFold on the "phantom generated" sequence to obtain the structure, to obtain scRMSD (this is done simply with `omegafold <file> <outdir>`, assuming you have OmegaFold installed).

### Generating Inverse and Phantom Generation

TODO: highlight what inverse and phantom means using a diagram

To calculate self-consistency and cross-consistency, we need to first generate inverse and phantom generations, i.e. steps 3 to 6 of the pipeline. This can be run together using `pipeline/run_consistency.py`. We use this script for all reported baselines.

```
python pipeline/run_consistency.py --help  # show all config options

python pipeline/run_consistency.py ++samples_dir=/data/lux70/plaid/artifacts/samples/5j007z42/compositional/f166_o818_l148_s3/f166_o818
```


To obtain metrics for analysis (e.g. ccRMSD, etc.), use `pipeline/run_analysis.py`:

```
python pipeline/run_analysis.py /data/lux70/plaid/artifacts/samples/5j007z42/compositional/f166_o818_l148_s3/f166_o818
```

To run the Foldseek pipeline, you can do:
```
python /homefs/home/robins21/scripts/run_foldseek_novelty_diversity2.py -input_folder /data/lux70/plaid/artifacts/samples/5j007z42/compositional/f166_o818_l148_s3/f166_o818/generated/structures -outputfolder /data/lux70/plaid/artifacts/samples/5j007z42/compositional/f166_o818_l148_s3/f166_o818fold_seek_results_/ -outputfile foldseek_filtered --d --n --f
```

We also provide scripts for sweeping through sampling hyperparmeters using SLURM. Scripts for running `pipeline/run_{step}.py` can be found at `scripts/eval/run_{step}.sh`.

#### Additional Evaluations

* `scripts/eval/loop_compositional.sh`: hyperparameter tunes across SLURM jobs and launches them concurrently.
* `scripts/eval/run_consistency.sh`: loops through a series of folders and runs consistency experiments
* `scripts/eval/sample_by_median_length.py`: 

These provide entry points for inference experiments at `configs/pipeline/experiments`.


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
