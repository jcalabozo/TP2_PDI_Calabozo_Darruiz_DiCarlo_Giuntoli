# Trabajo Práctico N° 2 — Procesamiento de Imágenes I

Tecnicatura Universitaria en Inteligencia Artificial · FCEIA · Universidad Nacional de Rosario
Procesamiento de Imágenes I (IA 4.4) · 2026, 2° semestre

**Integrantes:** Josías Calabozo, Ismael Darruiz, Sebastián Di Carlo, Sharo Giuntoli

El trabajo resuelve los dos problemas planteados en la [consigna](docs/TUIA_PDI_TP2_2026_C2.pdf):

1. **Detección y clasificación de monedas y dados:** segmentación de las monedas y los dados de `monedas.jpg`, clasificación y conteo de las monedas, y lectura del valor de cada dado.
2. **Detección de patentes:** detección de la patente en cada una de las doce imágenes y segmentación de sus caracteres.

## Estructura del repositorio

```text
.
├── src/                    Código fuente
├── datos/                  Imágenes y pistas provistas por la cátedra
│   ├── monedas.jpg
│   ├── Patentes/
│   │   └── img01.png … img12.png
│   ├── info_patentes.txt   Pistas para segmentar los caracteres de las patentes
│   └── Factor de Forma.txt Cómo calcular el factor de forma de un contorno
├── docs/                   Consigna e informe
│   └── TUIA_PDI_TP2_2026_C2.pdf
└── README.md
```

## Requisitos

- Python 3.14
- numpy 2.5.2
- opencv-contrib-python 5.0.0.93
- matplotlib 3.11.1

### Instalación

Desde la raíz del repositorio, en Git Bash (Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install numpy==2.5.2 opencv-contrib-python==5.0.0.93 matplotlib==3.11.1
```

En PowerShell, el entorno se activa con `.venv\Scripts\Activate.ps1`; en Linux o macOS, con `source .venv/bin/activate`.

## Ejecución

Pendiente: se completa con el comando de cada script cuando estén resueltos los dos problemas.
