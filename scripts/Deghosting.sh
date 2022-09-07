#!/bin/sh

# DeepPrec deghosting

# Syncline
python Deghosting.py -c ../config/Syncline_exp1.yaml

# Marmousi
python Deghosting.py -c ../config/Marmousi_exp1.yaml
