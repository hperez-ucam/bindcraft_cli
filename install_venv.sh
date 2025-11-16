#!/bin/bash
# BindCraft installation script for virtual environment
# This script creates a virtual environment and installs all possible dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=========================================="
echo "BindCraft Installation in Virtual Environment"
echo "=========================================="

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "1. Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created in $VENV_DIR"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "2. Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Update pip
echo ""
echo "3. Updating pip, setuptools and wheel..."
pip install --upgrade pip setuptools wheel

# Install compatible NumPy
echo ""
echo "4. Installing NumPy < 2.0 (compatible with pandas)..."
pip install 'numpy<2' pandas

# Install ColabDesign
echo ""
echo "5. Installing ColabDesign..."
pip install git+https://github.com/sokrypton/ColabDesign.git

# Install PyRosetta using pyrosetta-installer
echo ""
echo "6. Installing PyRosetta..."
echo "   (This may take several minutes, ~1.7 GB)"
pip install pyrosettacolabsetup
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
echo "✓ PyRosetta installed"

# Verify that bindcraft exists
echo ""
echo "7. Verifying BindCraft..."
if [ ! -d "$SCRIPT_DIR/bindcraft" ]; then
    echo "   Cloning BindCraft repository..."
    git clone https://github.com/martinpacesa/BindCraft.git "$SCRIPT_DIR/bindcraft"
    chmod +x "$SCRIPT_DIR/bindcraft/functions/dssp"
    chmod +x "$SCRIPT_DIR/bindcraft/functions/DAlphaBall.gcc"
    echo "✓ BindCraft cloned"
else
    echo "✓ BindCraft already exists"
fi

# Verify AlphaFold2 parameters
echo ""
echo "8. Verifying AlphaFold2 parameters..."
if [ ! -f "$SCRIPT_DIR/bindcraft/params/done.txt" ]; then
    echo "   ⚠️  AlphaFold2 parameters not found"
    echo "   To download them, run:"
    echo "   cd $SCRIPT_DIR/bindcraft/params"
    echo "   aria2c -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    echo "   tar -xf alphafold_params_2022-12-06.tar"
    echo "   touch done.txt"
else
    echo "✓ AlphaFold2 parameters found"
fi

# Check GPU
echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: GPU Requirement"
echo "=========================================="
echo ""
echo "BindCraft requires a CUDA-compatible GPU to run."
echo "Without GPU, the script can load but will not execute the pipeline."
echo ""
echo "To install JAX with CUDA support:"
echo "  pip install --upgrade 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
echo ""

# Verify installation
echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate the virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run BindCraft:"
echo "  source $VENV_DIR/bin/activate"
echo "  python bindcraft_cli.py --config config.json"
echo ""
echo "=========================================="

