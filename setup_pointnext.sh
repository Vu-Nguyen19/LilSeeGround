#!/usr/bin/env bash
# Script to build PointNeXt C++ extensions

# Make sure you are in the project root and the conda environment is activated.

echo "Initializing git submodules..."
git submodule update --init --recursive

echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

echo "Building PointNeXt C++ extensions..."

# Path to PointNeXt submodule
POINTNEXT_DIR="./models/pointnext/PointNeXt"

# 1. Pointnet2 batch
echo "1. Building pointnet2_batch..."
cd "${POINTNEXT_DIR}/openpoints/cpp/pointnet2_batch"
python setup.py install
cd -

# 2. Grid subsampling
echo "2. Building grid subsampling..."
cd "${POINTNEXT_DIR}/openpoints/cpp/subsampling"
python setup.py build_ext --inplace
cd -

# 3. Pointops
echo "3. Building pointops..."
cd "${POINTNEXT_DIR}/openpoints/cpp/pointops"
python setup.py install
cd -

# 4. Chamfer distance (optional)
echo "4. Building chamfer_dist..."
cd "${POINTNEXT_DIR}/openpoints/cpp/chamfer_dist"
python setup.py install
cd -

# 5. EMD (optional)
echo "5. Building emd..."
cd "${POINTNEXT_DIR}/openpoints/cpp/emd"
python setup.py install
cd -

echo "All PointNeXt extensions built successfully!"
