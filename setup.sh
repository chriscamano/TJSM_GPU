# Navigate into the code directory
cd "$(dirname "$0")/code"
# Create and enter the build directory
mkdir -p build
cd build
# Run CMake to configure the build
cmake ..
# Return to the code directory and install dependencies in editable mode
cd ..
pip install -e .