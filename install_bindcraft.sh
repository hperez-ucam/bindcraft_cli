#!/bin/bash
# Script de instalación de BindCraft para entorno CLI
# Este script instala todas las dependencias necesarias para ejecutar BindCraft

set -e  # Salir si hay algún error

echo "=========================================="
echo "Instalación de BindCraft"
echo "=========================================="

# Directorio base (donde se ejecuta el script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDCRAFT_DIR="$SCRIPT_DIR/bindcraft"

# Verificar si BindCraft ya está instalado
if [ -f "$BINDCRAFT_DIR/params/done.txt" ]; then
    echo "✓ BindCraft ya está instalado en $BINDCRAFT_DIR"
    echo "Si quieres reinstalar, elimina el directorio bindcraft/ primero"
    exit 0
fi

echo ""
echo "1. Clonando repositorio de BindCraft..."
if [ ! -d "$BINDCRAFT_DIR" ]; then
    git clone https://github.com/martinpacesa/BindCraft.git "$BINDCRAFT_DIR"
    echo "✓ Repositorio clonado"
else
    echo "✓ Directorio bindcraft/ ya existe"
fi

# Dar permisos de ejecución a los binarios necesarios
echo ""
echo "2. Configurando permisos de ejecución..."
if [ -f "$BINDCRAFT_DIR/functions/dssp" ]; then
    chmod +x "$BINDCRAFT_DIR/functions/dssp"
    echo "✓ Permisos configurados para dssp"
fi

if [ -f "$BINDCRAFT_DIR/functions/DAlphaBall.gcc" ]; then
    chmod +x "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
    echo "✓ Permisos configurados para DAlphaBall.gcc"
fi

# Instalar ColabDesign
echo ""
echo "3. Instalando ColabDesign..."
pip install git+https://github.com/sokrypton/ColabDesign.git
echo "✓ ColabDesign instalado"

# Descargar parámetros de AlphaFold2
echo ""
echo "4. Descargando parámetros de AlphaFold2..."
echo "   (Esto puede tardar varios minutos, ~4GB de datos)"

PARAMS_DIR="$BINDCRAFT_DIR/params"
mkdir -p "$PARAMS_DIR"

# Verificar si aria2 está instalado
if ! command -v aria2c &> /dev/null; then
    echo "   Instalando aria2 para descarga más rápida..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y aria2
    elif command -v yum &> /dev/null; then
        sudo yum install -y aria2
    elif command -v brew &> /dev/null; then
        brew install aria2
    else
        echo "   ⚠ aria2 no está disponible. Usando wget/curl (más lento)..."
        DOWNLOAD_CMD="wget"
    fi
fi

if command -v aria2c &> /dev/null; then
    cd "$PARAMS_DIR"
    aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    tar -xf alphafold_params_2022-12-06.tar
    rm alphafold_params_2022-12-06.tar
    touch done.txt
    echo "✓ Parámetros de AlphaFold2 descargados"
else
    echo "   ⚠ Por favor, descarga manualmente los parámetros de AlphaFold2:"
    echo "   URL: https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    echo "   Extrae el archivo en: $PARAMS_DIR"
    echo "   Luego ejecuta: touch $PARAMS_DIR/done.txt"
fi

# Instalar PyRosetta
echo ""
echo "5. Instalando PyRosetta..."
echo "   (Esto puede tardar varios minutos)"

pip install pyrosettacolabsetup

# Verificar instalación de PyRosetta
python3 << EOF
import sys
import contextlib
import io

try:
    import pyrosettacolabsetup
    with contextlib.redirect_stdout(io.StringIO()):
        pyrosettacolabsetup.install_pyrosetta(serialization=True, cache_wheel_on_google_drive=False)
    print("✓ PyRosetta instalado correctamente")
except Exception as e:
    print(f"✗ Error instalando PyRosetta: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✓ Instalación completada!"
echo "=========================================="
echo ""
echo "Próximos pasos:"
echo "1. Asegúrate de tener una GPU compatible con JAX"
echo "2. Configura tu archivo config.json"
echo "3. Ejecuta: python bindcraft_cli.py --config config.json"
echo ""


