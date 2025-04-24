# Exit on error
set -e
# Define environment name
ENV_NAME="tjsm_env"
# Create conda environment with Python 3.8 and necessary packages
conda create -y -n "$ENV_NAME" python=3.8 numpy matplotlib
# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
# Navigate into the code directory
cd "$(dirname "$0")/code"
# Create and enter the build directory
mkdir -p build
cd build
# Run CMake to configure the build
cmake ..
# Build the project
make
# Retrn to the code directory
cd ..
# Install Python package in editable mode
pip install -e .