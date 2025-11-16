#!/bin/bash
# BindCraft installation script for CLI environment
# This script installs all necessary dependencies to run BindCraft

set -e  # Exit on any error

echo "=========================================="
echo "BindCraft Installation"
echo "=========================================="

# Base directory (where the script is executed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDCRAFT_DIR="$SCRIPT_DIR/bindcraft"

# Check if BindCraft is already installed
if [ -f "$BINDCRAFT_DIR/params/done.txt" ]; then
    echo "✓ BindCraft is already installed in $BINDCRAFT_DIR"
    echo "If you want to reinstall, delete the bindcraft/ directory first"
    exit 0
fi

echo ""
echo "1. Cloning BindCraft repository..."
if [ ! -d "$BINDCRAFT_DIR" ]; then
    git clone https://github.com/martinpacesa/BindCraft.git "$BINDCRAFT_DIR"
    echo "✓ Repository cloned"
else
    echo "✓ bindcraft/ directory already exists"
fi

# Set execution permissions for necessary binaries
echo ""
echo "2. Setting execution permissions..."
if [ -f "$BINDCRAFT_DIR/functions/dssp" ]; then
    chmod +x "$BINDCRAFT_DIR/functions/dssp"
    echo "✓ Permissions set for dssp"
fi

if [ -f "$BINDCRAFT_DIR/functions/DAlphaBall.gcc" ]; then
    chmod +x "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
    echo "✓ Permissions set for DAlphaBall.gcc"
fi

# Install ColabDesign
echo ""
echo "3. Installing ColabDesign..."
pip install git+https://github.com/sokrypton/ColabDesign.git
echo "✓ ColabDesign installed"

# Download AlphaFold2 parameters
echo ""
echo "4. Downloading AlphaFold2 parameters..."
echo "   (This may take several minutes, ~4GB of data)"

PARAMS_DIR="$BINDCRAFT_DIR/params"
mkdir -p "$PARAMS_DIR"

# Check if aria2 is installed
if ! command -v aria2c &> /dev/null; then
    echo "   Installing aria2 for faster download..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y aria2
    elif command -v yum &> /dev/null; then
        sudo yum install -y aria2
    elif command -v brew &> /dev/null; then
        brew install aria2
    else
        echo "   ⚠ aria2 is not available. Using wget/curl (slower)..."
        DOWNLOAD_CMD="wget"
    fi
fi

if command -v aria2c &> /dev/null; then
    cd "$PARAMS_DIR"
    aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    tar -xf alphafold_params_2022-12-06.tar
    rm alphafold_params_2022-12-06.tar
    touch done.txt
    echo "✓ AlphaFold2 parameters downloaded"
else
    echo "   ⚠ Please manually download AlphaFold2 parameters:"
    echo "   URL: https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    echo "   Extract the file to: $PARAMS_DIR"
    echo "   Then run: touch $PARAMS_DIR/done.txt"
fi

# Install PyRosetta
echo ""
echo "5. Installing PyRosetta..."
echo "   (This may take several minutes)"

pip install pyrosettacolabsetup

# Verify PyRosetta installation
python3 << EOF
import sys
import contextlib
import io

try:
    import pyrosettacolabsetup
    with contextlib.redirect_stdout(io.StringIO()):
        pyrosettacolabsetup.install_pyrosetta(serialization=True, cache_wheel_on_google_drive=False)
    print("✓ PyRosetta installed successfully")
except Exception as e:
    print(f"✗ Error installing PyRosetta: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✓ Installation completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure you have a JAX-compatible GPU"
echo "2. Configure your config.json file"
echo "3. Run: python bindcraft_cli.py --config config.json"
echo ""

