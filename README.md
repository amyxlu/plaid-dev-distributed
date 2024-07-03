# PLAID (Protein LAtent Induced Diffusion)

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