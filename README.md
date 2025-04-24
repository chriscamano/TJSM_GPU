# Tracial Joint Spectral Measure Visualization (GPU Support)

The code numerically visualizes the **[tracial joint spectral measure](https://arxiv.org/pdf/2310.03227)** \( \mu_{\mathbf{A}, \mathbf{B}} \) associated to the Hermitian matrices  

\[
\mathbf{A} = \frac{\mathbf{C} + \mathbf{C}^*}{2}, \quad \mathbf{B} = \frac{\mathbf{C} - \mathbf{C}^*}{2i},
\]

by computing its continuous density  

\[
\rho(a, b) = \frac{1}{2\pi} \sum_{j=1}^n \left| \operatorname{Im} \lambda_j\left( \left( \frac{\mathbf{I} - \frac{a}{a^2 + b^2} \mathbf{A} + \frac{b}{a^2 + b^2} \mathbf{B}}{1} \right)(b\mathbf{A} - a\mathbf{B})^{-1} \right) \right|.
\]

This density describes how linear combinations \( x\mathbf{A} + y\mathbf{B} \) contribute spectrally across \( (a,b) \in \mathbb{R}^2 \). Singular components supported along rays through eigenvectors of \( \mathbf{A}^{-1} \mathbf{B} \) are overlaid via their Rayleigh quotients \( (\langle \mathbf{A}v, v \rangle, \langle \mathbf{B}v, v \rangle) \).

---

## Requirements

- **Compiler**: `clang` ≥ 10.0  
- **Build system**: `CMake` ≥ 3.12  
- **Dependencies**:  
  - Header-only [Eigen](https://eigen.tuxfamily.org/)  
  - Python 3.8+ with `numpy`, `matplotlib`  
- **Conda** (for managing the Python environment)

---

## Setup

Before running the setup script, ensure that [Conda](https://docs.conda.io/en/latest/miniconda.html) is installed and available in your terminal. You can check this with:

```bash
conda --version
```

From the project root, run:
```bash
chmod +x setup.sh
./setup.sh
```
This script will:
- Check that conda is installed
- Create and activate a new environment called tjsm_env
- Install required Python packages (numpy, matplotlib)
- Build the C++ backend with cmake and make
- Install the Python package in editable mode with pip install -e .

⚠️ Note: This setup is primarily tested on Unix-like systems (Linux/macOS). Windows users may need to adapt the script or use WSL (Windows Subsystem for Linux).

⸻
## Examples
Example experiments that use the C++ and `torch` implementations can be found in the `experiments` directory.
