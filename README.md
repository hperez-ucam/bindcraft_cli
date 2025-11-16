# BindCraft CLI

Este es un script de línea de comandos para ejecutar el pipeline de diseño de proteínas BindCraft, originalmente diseñado para Google Colab.

## Requisitos

1. Python 3.7+
2. BindCraft instalado (ver instalación en el notebook original)
3. Todas las dependencias de BindCraft (ColabDesign, PyRosetta, etc.)

## Instalación

### Opción 1: Instalación Automática (Recomendada)

Ejecuta el script de instalación que instala todas las dependencias automáticamente:

```bash
./install_bindcraft.sh
```

Este script:
- Clona el repositorio de BindCraft
- Instala ColabDesign
- Descarga los parámetros de AlphaFold2 (~4GB)
- Instala PyRosetta
- Configura todos los permisos necesarios

**Nota**: La instalación puede tardar varios minutos, especialmente la descarga de los parámetros de AlphaFold2.

### Opción 2: Instalación Manual

Si prefieres instalar manualmente:

1. Clonar el repositorio de BindCraft:
```bash
git clone https://github.com/martinpacesa/BindCraft.git bindcraft
```

2. Instalar las dependencias:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git
pip install pyrosettacolabsetup
```

3. Descargar los parámetros de AlphaFold2:
```bash
mkdir -p bindcraft/params
cd bindcraft/params
aria2c -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar
rm alphafold_params_2022-12-06.tar
touch done.txt
```

4. Configurar permisos:
```bash
chmod +x bindcraft/functions/dssp
chmod +x bindcraft/functions/DAlphaBall.gcc
```

## Uso

### 1. Crear archivo de configuración

Crea un archivo JSON con los parámetros de diseño. Puedes usar `config.json` como plantilla:

```json
{
  "design_path": "./output/PDL1",
  "binder_name": "PDL1",
  "starting_pdb": "./example/PDL1.pdb",
  "chains": "A",
  "target_hotspot_residues": "",
  "lengths": [70, 150],
  "number_of_final_designs": 100,
  "load_previous_target_settings": "",
  "design_protocol": "Default",
  "prediction_protocol": "Default",
  "interface_protocol": "AlphaFold2",
  "template_protocol": "Default",
  "filter_option": "Default"
}
```

### 2. Ejecutar el pipeline

**Opción A: Usando el script wrapper (Recomendado)**

```bash
./run_bindcraft.sh --config config.json
```

Este script activa automáticamente el entorno virtual y verifica que todas las dependencias estén disponibles.

**Opción B: Activando el entorno virtual manualmente**

```bash
source venv/bin/activate
python bindcraft_cli.py --config config.json
```

**Nota importante**: Debes usar el Python del entorno virtual (`venv/`) donde están instaladas todas las dependencias (PyRosetta, ColabDesign, etc.). Si ejecutas `python` directamente sin activar el venv, obtendrás errores de importación.

## Parámetros de configuración

### Parámetros básicos

- **design_path**: Ruta donde se guardarán los diseños generados
- **binder_name**: Nombre que se usará como prefijo para los binders diseñados
- **starting_pdb**: Ruta al archivo PDB de la proteína objetivo
- **chains**: Cadenas a targetear (ej: "A" o "A,C")
- **target_hotspot_residues**: Posiciones específicas a targetear (opcional, ej: "1,2-10" o "A1-10,B1-20")
- **lengths**: Rango de longitudes del binder [min, max]
- **number_of_final_designs**: Número de diseños finales requeridos
- **load_previous_target_settings**: Ruta a configuración previa para continuar un diseño (opcional)

### Parámetros avanzados

- **design_protocol**: Protocolo de diseño
  - `"Default"`: Protocolo por defecto recomendado
  - `"Beta-sheet"`: Promueve más estructuras de lámina beta
  - `"Peptide"`: Optimizado para binders peptídicos helicoidales

- **prediction_protocol**: Protocolo de predicción
  - `"Default"`: Predicción de secuencia única del binder
  - `"HardTarget"`: Usa conjetura inicial para mejorar predicción de complejos difíciles

- **interface_protocol**: Método de diseño de interfaz
  - `"AlphaFold2"`: Interfaz generada por AlphaFold2 (por defecto)
  - `"MPNN"`: Usa MPNN soluble para optimizar la interfaz

- **template_protocol**: Protocolo de template del objetivo
  - `"Default"`: Permite flexibilidad limitada
  - `"Masked"`: Permite mayor flexibilidad del objetivo a nivel de cadena lateral y backbone

### Filtros

- **filter_option**: Opción de filtros para los diseños
  - `"Default"`: Filtros recomendados
  - `"Peptide"`: Para diseño de binders peptídicos
  - `"Relaxed"`: Más permisivos pero pueden resultar en menos éxitos experimentales
  - `"Peptide_Relaxed"`: Más permisivos para péptidos no helicoidales
  - `"None"`: Sin filtros (para benchmarking)

## Estructura de salida

El pipeline genera la siguiente estructura de directorios:

```
design_path/
├── Trajectory/          # Trajectorias generadas
├── Trajectory/Relaxed/  # Trajectorias relajadas
├── MPNN/                # Diseños MPNN
├── MPNN/Relaxed/        # Diseños MPNN relajados
├── Accepted/            # Diseños aceptados
├── Rejected/            # Diseños rechazados
├── trajectory_stats.csv # Estadísticas de trajectorias
├── mpnn_design_stats.csv # Estadísticas de diseños MPNN
└── final_design_stats.csv # Estadísticas finales
```

## Notas

- **IMPORTANTE**: Siempre usa el script `run_bindcraft.sh` o activa el entorno virtual antes de ejecutar. El Python del sistema no tiene las dependencias instaladas.
- El script requiere que BindCraft esté instalado y configurado correctamente
- Se necesita una GPU compatible con JAX para ejecutar el pipeline
- El script puede continuar desde donde se quedó si se interrumpe, siempre que los archivos CSV existan
- Para continuar un diseño previo, usa `load_previous_target_settings` apuntando al archivo JSON generado anteriormente

## Diferencias con el notebook de Colab

- No requiere montar Google Drive
- No usa widgets de IPython para visualización en tiempo real
- Los contadores de progreso se muestran en la consola
- La configuración se lee desde un archivo JSON en lugar de widgets

