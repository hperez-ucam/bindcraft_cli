#!/bin/bash
# Script de instalación de BindCraft en entorno virtual
# Este script crea un entorno virtual e instala todas las dependencias posibles

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=========================================="
echo "Instalación de BindCraft en Entorno Virtual"
echo "=========================================="

# Crear entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "1. Creando entorno virtual..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Entorno virtual creado en $VENV_DIR"
else
    echo "✓ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo ""
echo "2. Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# Actualizar pip
echo ""
echo "3. Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel

# Instalar NumPy compatible
echo ""
echo "4. Instalando NumPy < 2.0 (compatible con pandas)..."
pip install 'numpy<2' pandas

# Instalar ColabDesign
echo ""
echo "5. Instalando ColabDesign..."
pip install git+https://github.com/sokrypton/ColabDesign.git

# Instalar PyRosetta usando pyrosetta-installer
echo ""
echo "6. Instalando PyRosetta..."
echo "   (Esto puede tardar varios minutos, ~1.7 GB)"
pip install pyrosettacolabsetup
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
echo "✓ PyRosetta instalado"

# Verificar que bindcraft existe
echo ""
echo "7. Verificando BindCraft..."
if [ ! -d "$SCRIPT_DIR/bindcraft" ]; then
    echo "   Clonando repositorio de BindCraft..."
    git clone https://github.com/martinpacesa/BindCraft.git "$SCRIPT_DIR/bindcraft"
    chmod +x "$SCRIPT_DIR/bindcraft/functions/dssp"
    chmod +x "$SCRIPT_DIR/bindcraft/functions/DAlphaBall.gcc"
    echo "✓ BindCraft clonado"
else
    echo "✓ BindCraft ya existe"
fi

# Verificar parámetros de AlphaFold2
echo ""
echo "8. Verificando parámetros de AlphaFold2..."
if [ ! -f "$SCRIPT_DIR/bindcraft/params/done.txt" ]; then
    echo "   ⚠️  Parámetros de AlphaFold2 no encontrados"
    echo "   Para descargarlos, ejecuta:"
    echo "   cd $SCRIPT_DIR/bindcraft/params"
    echo "   aria2c -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    echo "   tar -xf alphafold_params_2022-12-06.tar"
    echo "   touch done.txt"
else
    echo "✓ Parámetros de AlphaFold2 encontrados"
fi

# Verificar GPU
echo ""
echo "=========================================="
echo "⚠️  IMPORTANTE: Requisito de GPU"
echo "=========================================="
echo ""
echo "BindCraft requiere una GPU compatible con CUDA para ejecutarse."
echo "Sin GPU, el script puede cargar pero no ejecutar el pipeline."
echo ""
echo "Para instalar JAX con soporte CUDA:"
echo "  pip install --upgrade 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
echo ""

# Verificar instalación
echo ""
echo "=========================================="
echo "Resumen de Instalación"
echo "=========================================="
echo ""
echo "Entorno virtual: $VENV_DIR"
echo ""
echo "Para activar el entorno virtual:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Para ejecutar BindCraft:"
echo "  source $VENV_DIR/bin/activate"
echo "  python bindcraft_cli.py --config config.json"
echo ""
echo "=========================================="

