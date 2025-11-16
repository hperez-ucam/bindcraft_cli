#!/bin/bash
# Wrapper script to run BindCraft CLI with virtual environment activated

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found in: $SCRIPT_DIR/venv"
    echo ""
    echo "Please run first:"
    echo "  ./install_venv.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if PyRosetta is available
if ! python -c "import pyrosetta" 2>/dev/null; then
    echo "❌ Error: PyRosetta is not available in the virtual environment"
    echo ""
    echo "Please run:"
    echo "  source venv/bin/activate"
    echo "  pip install pyrosetta-installer"
    echo "  pyrosetta-installer --install"
    exit 1
fi

# Run the CLI script with all passed arguments
python bindcraft_cli.py "$@"
