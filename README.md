# PLAID (Protein LAtent Induced Diffusion)

Install:
```
conda create --file environment.yaml
conda activate plaid
```

OpenFold should be installed individually to trigger CUDA builds without downloading other dependencies:
```
git clone https://github.com/amyxlu/openfold.git
cd openfold
python setup.py develop
```
