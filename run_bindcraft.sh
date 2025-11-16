#!/bin/bash
# Wrapper script para ejecutar BindCraft CLI con el entorno virtual activado

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verificar que el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Error: Entorno virtual no encontrado en: $SCRIPT_DIR/venv"
    echo ""
    echo "Por favor, ejecuta primero:"
    echo "  ./install_venv.sh"
    exit 1
fi

# Activar el entorno virtual
source venv/bin/activate

# Verificar que PyRosetta está disponible
if ! python -c "import pyrosetta" 2>/dev/null; then
    echo "❌ Error: PyRosetta no está disponible en el entorno virtual"
    echo ""
    echo "Por favor, ejecuta:"
    echo "  source venv/bin/activate"
    echo "  pip install pyrosetta-installer"
    echo "  pyrosetta-installer --install"
    exit 1
fi

# Ejecutar el script CLI con todos los argumentos pasados
python bindcraft_cli.py "$@"

