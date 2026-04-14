#!/usr/bin/env bash
# Complete script to set up the SeeGround environment

# 1. Create the conda environment
echo "Creating conda environment 'seeground'..."
conda env create -f environment.yml

# 2. Activate the environment (need to use source because it's in a script)
# This might not work in all shell scripts depending on how conda is set up.
# Better to tell the user to activate it and run the second part.
echo "Environment 'seeground' created."
echo "Please run the following commands to complete the setup:"
echo "    conda activate seeground"
echo "    ./setup_pointnext.sh"

# If you want to try to automate activation:
# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate seeground
# ./setup_pointnext.sh
