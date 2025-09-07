#!/bin/bash
set -ex

# Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda config --set ssl_verify no
# Create Conda Environment
echo "conda create -n fedml2025 python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
conda activate fedml

# Install Pyflakes within Conda environment
echo "Installing pyflakes"
conda install -c conda-forge pyflakes

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines)
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required packages
conda install scikit-learn
conda install numpy
conda install h5py
conda install setproctitle
conda install networkx
conda install tqdm

# Install additional requirements from requirements.txt
pip install -r requirements.txt
