# Exit on error
set -e

# Define environment name
ENV_NAME="tjsm_env"

# Create conda environment with Python 3.12 and necessary packages
conda create -y -n "$ENV_NAME" python=3.12 numpy matplotlib

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Ensure we have the correct libstdc++ symbols
conda install -y -c conda-forge libgcc-ng libstdcxx-ng

# Navigate into the code directory
cd "$(dirname "$0")/code"

# Create and enter the build directory
mkdir -p build
cd build

# Run CMake to configure the build
cmake ..

# Build the project
make

# Return to the project root (where setup.py lives)
cd ../..

# Install Python package in editable mode
pip install -e .