#!/bin/bash
# 
# Installer for deepprecs environment
# 
# Run: ./install.sh
# 
# M. Ravasi, 14/08/2022

echo 'Creating deepprecs environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepprecs
conda env list
echo 'Created and activated environment:' $(which python)

# remove numba (temporary, due to collision with numpy version)
rm -rf /home/ravasim/miniconda3/envs/deepprecs/lib/python3.10/site-packages/numba*

# install deepprecs package
pip install -e .

# check cupy-torch work as expected
echo 'Checking cupy version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

