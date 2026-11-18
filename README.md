# Trabajo Práctico N° 2 — Procesamiento de Imágenes I

Tecnicatura Universitaria en Inteligencia Artificial · FCEIA · Universidad Nacional de Rosario  
Procesamiento de Imágenes I (IA 4.4) · 2026, 2° semestre

**Integrantes:** Josías Calabozo, Ismael Darruiz, Sebastián Di Carlo, Sharo Giuntoli

El trabajo resuelve los dos problemas de la [consigna](docs/TUIA_PDI_TP2_2026_C2.pdf): detección y conteo de monedas/dados, y detección de patentes con segmentación de caracteres en las doce imágenes de la carpeta `datos/Patentes`.

## Estructura

```text
.
├── src/          Código fuente
├── datos/        Imágenes y pistas de la cátedra
├── resultados/   Salidas generadas por los scripts
├── docs/         Consigna e informe
└── README.md
```

## Requisitos e instalación

- Python 3.14
- numpy 2.5.2
- opencv-contrib-python 5.0.0.93
- matplotlib 3.11.1

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install numpy==2.5.2 opencv-contrib-python==5.0.0.93 matplotlib==3.11.1
```

En PowerShell, activar con `.venv\Scripts\Activate.ps1`.

## Ejecución

Desde la raíz del repositorio:

```bash
python src/problema1_monedas_dados.py
python src/problema2_patentes.py
```

Los scripts leen `datos/` y generan `resultados/`. El primero muestra en pantalla, imprime y guarda cada etapa: el análisis del fondo, el intento con top-hat, los bordes, la máscara (con el detalle de una moneda en cada paso), la separación entre monedas y dados, el área de cada moneda, los puntos de cada dado y el resultado final. El segundo muestra primero el detalle de cada paso con `img01.png` (binarización, candidatos a carácter, elección del umbral, placa y placa enderezada) y después procesa las doce imágenes: imprime los resultados de cada una y muestra una figura por etapa con las doce (largo de la cadena según el umbral, candidatos, placas detectadas, caracteres y placas enderezadas). En los dos scripts, cada figura se muestra en una ventana y el script sigue al cerrarla.

## Documentación

- [Informe PDF](docs/informe.pdf)
- [Fuente del informe](docs/informe.md)
