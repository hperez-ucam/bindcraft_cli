#!/bin/bash
# Script interactivo para hacer push a GitHub

cd /home/horacio/bindcraft

echo "=== Push a GitHub ==="
echo ""
echo "Este script te pedirá tus credenciales de GitHub"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "bindcraft_cli.py" ]; then
    echo "Error: No estás en el directorio correcto"
    exit 1
fi

# Verificar estado
echo "Estado del repositorio:"
git status --short | head -5
echo ""

# Pedir credenciales
read -p "GitHub Username (hperez-ucam): " GIT_USER
GIT_USER=${GIT_USER:-hperez-ucam}

echo ""
echo "Para la contraseña, necesitas un Personal Access Token."
echo "Si no tienes uno, créalo en: https://github.com/settings/tokens"
echo ""
read -s -p "GitHub Personal Access Token: " GIT_TOKEN
echo ""

if [ -z "$GIT_TOKEN" ]; then
    echo "Error: Token requerido"
    exit 1
fi

# Configurar URL con credenciales
GIT_URL="https://${GIT_USER}:${GIT_TOKEN}@github.com/hperez-ucam/bindcraft_cli.git"

echo ""
echo "Haciendo push..."
git push $GIT_URL main 2>&1

# Limpiar credenciales de la URL del remoto
git remote set-url origin https://github.com/hperez-ucam/bindcraft_cli.git

echo ""
echo "✅ Push completado (o verifica errores arriba)"

