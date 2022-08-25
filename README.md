![LOGO](https://github.com/DIG-Kaust/DeepPrecs/blob/main/logo.png)

``deepprecs`` is a Python library aimed at providing the fundamental building blocks to solve Inverse Problems with 
the aid of Deep Preconditioners.

For more details refer to the accompanying paper: **Deep Preconditioners and their application to seismic wavefield processing -
Ravasi M.** submitted to Frontiers in Earth Science.

## Project structure
This repository is organized as follows:

* :open_file_folder: **deepprecs**: python library containing routines for training and application of deep preconditioners to inverse problems;
* :open_file_folder: **data**: folder containing instructions on how to retrieve the data used in the examples;
* :open_file_folder: **notebooks**: set of jupyter notebooks applying deep preconditioners to a number of problems of increasing complexity;
* :open_file_folder: **scripts**: set of python scripts used to run seismic interpolation experiments in batch mode.

## Notebooks
The following notebooks are provided:

- :orange_book: ``X1.ipynb``: notebook performing ...;
- :orange_book: ``X2.ipynb``: notebook performing ...


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate deepprecs
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
