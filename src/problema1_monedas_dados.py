"""Problema 1: segmentación, clasificación y conteo de monedas y dados."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
IMAGEN = Path("datos") / "monedas.jpg"
CARPETA_SALIDA = Path("resultados") / "problema1"

# Segmentación por bordes.
SIGMA = 2  # desvío del suavizado gaussiano antes de Canny
CANNY_BAJO = 20
CANNY_ALTO = 60
LARGO_MINIMO = 50  # en píxeles; los bordes más cortos son textura del tapete
LADO_CIERRE = 35  # elemento estructural que cierra los contornos abiertos
LADO_APERTURA = 41  # elemento estructural que limpia la máscara

# Clasificación.
RHO_UMBRAL = 0.8  # factor de forma: por debajo es un dado
SALTO_AREA = 1.10  # un área 10 % mayor que la anterior empieza otro tipo de moneda
NOMBRES = ["10 centavos", "1 peso", "50 centavos"]  # de menor a mayor tamaño
AREA_PUNTO = 0.02  # área mínima de un punto del dado, como fracción del área del dado

# Colores en RGB para dibujar sobre la imagen.
AZUL = (30, 90, 255)
ROJO = (230, 30, 30)
VERDE = (0, 200, 0)
COLORES_TIPO = [(0, 190, 255), (255, 0, 200), (255, 150, 0)]


def imshow(img, title=None, color_img=False, ticks=False):
    """Muestra una imagen en la figura actual, como la función imshow de la Unidad 3."""
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])


def rellenar_huecos(img):
    """Rellena los huecos de una máscara binaria (imfillhole_v2 de la Unidad 6)."""
    img_inundada = img.copy()
    alto, ancho = img.shape
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)
    # Inunda el fondo desde la esquina: lo que queda sin inundar son los huecos.
    cv2.floodFill(img_inundada, mascara, (0, 0), 255)
    return img | cv2.bitwise_not(img_inundada)


# --- Imagen original y fondo ----------------------------------------------------
img = cv2.imread(str(IMAGEN))
if img is None:
    raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGEN}")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
alto, ancho = gris.shape
print(f"Tamaño: {ancho} x {alto} píxeles")

# Las columnas de los bordes izquierdo y derecho son solo tapete.
print(f"Gris medio de las 100 columnas de la izquierda: {gris[:, :100].mean():.0f}")
print(f"Gris medio de las 100 columnas de la derecha: {gris[:, -100:].mean():.0f}")

umbral_otsu, mascara_otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Umbral de Otsu sobre toda la imagen: {umbral_otsu:.0f}")

plt.figure(figsize=(15, 4))
plt.subplot(131)
imshow(img_rgb, title="Imagen original", color_img=True, ticks=True)
plt.subplot(132)
plt.hist(gris.flatten(), bins=256, range=(0, 256))
plt.axvline(umbral_otsu, color="red", label=f"Otsu = {umbral_otsu:.0f}")
plt.legend()
plt.title("Histograma de grises")
plt.subplot(133)
imshow(mascara_otsu, title="Umbral de Otsu global")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "01_fondo_y_otsu.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Intento con top-hat ----------------------------------------------------------
# Unidad 6: la apertura con un elemento más grande que los objetos estima el fondo,
# y restarla deja un fondo uniforme. El elemento tiene que superar a la moneda más
# grande, de unos 360 píxeles de diámetro. Tarda unos segundos por su tamaño.
elemento_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (401, 401))
fondo_estimado = cv2.morphologyEx(gris, cv2.MORPH_OPEN, elemento_grande)
top_hat = cv2.subtract(gris, fondo_estimado)
umbral_th, mascara_th = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mascara_th = rellenar_huecos(mascara_th)

contornos_th, _ = cv2.findContours(mascara_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area_mayor_th = max(cv2.contourArea(c) for c in contornos_th)
print(f"\nTop-hat: umbral de Otsu {umbral_th:.0f}, {len(contornos_th)} regiones")
print(f"Top-hat: la región más grande tiene {area_mayor_th:.0f} píxeles")

plt.figure(figsize=(15, 4))
plt.subplot(131)
imshow(fondo_estimado, title="Fondo estimado (apertura 401x401)")
plt.subplot(132)
imshow(top_hat, title="Top-hat")
plt.subplot(133)
imshow(mascara_th, title=f"Top-hat umbralado (Otsu = {umbral_th:.0f}) y relleno")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "02_intento_top_hat.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Bordes ---------------------------------------------------------------------
suavizada = cv2.GaussianBlur(gris, (0, 0), SIGMA)
bordes = cv2.Canny(suavizada, CANNY_BAJO, CANNY_ALTO)

# Cada tramo de borde es una componente conectada; se descartan los cortos.
n_tramos, etiquetas, stats, _ = cv2.connectedComponentsWithStats(bordes, connectivity=8)
tramos_largos = [i for i in range(1, n_tramos) if stats[i, cv2.CC_STAT_AREA] >= LARGO_MINIMO]
bordes_largos = np.isin(etiquetas, tramos_largos).astype(np.uint8) * 255
print(f"\nCanny: {n_tramos - 1} tramos de borde, {len(tramos_largos)} de {LARGO_MINIMO} píxeles o más")

# Se muestra la zona central, donde el tapete tiene más textura, porque a escala
# de la imagen completa los bordes de un píxel no se ven.
zona = (slice(500, 1500), slice(1700, 3000))
plt.figure(figsize=(15, 4))
plt.subplot(131)
imshow(gris[zona], title="Zona central (gris)")
plt.subplot(132)
imshow(bordes[zona], title=f"Canny ({CANNY_BAJO}, {CANNY_ALTO})")
plt.subplot(133)
imshow(bordes_largos[zona], title=f"Tramos de {LARGO_MINIMO} píxeles o más")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "03_bordes.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Máscara de los objetos -------------------------------------------------------
elemento_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LADO_CIERRE, LADO_CIERRE))
dilatada = cv2.dilate(bordes_largos, elemento_cierre)
rellena = rellenar_huecos(dilatada)
# La erosión con el mismo elemento devuelve a los objetos su tamaño.
erosionada = cv2.erode(rellena, elemento_cierre)
elemento_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LADO_APERTURA, LADO_APERTURA))
mascara = cv2.morphologyEx(erosionada, cv2.MORPH_OPEN, elemento_apertura)

contornos_erosion, _ = cv2.findContours(erosionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Regiones después de la erosión: {len(contornos_erosion)}")
print(f"Regiones después de la apertura: {len(contornos)}")

plt.figure(figsize=(12, 8))
plt.subplot(221)
imshow(dilatada, title=f"Dilatación {LADO_CIERRE}x{LADO_CIERRE}")
plt.subplot(222)
imshow(rellena, title="Relleno de huecos")
plt.subplot(223)
imshow(erosionada, title=f"Erosión {LADO_CIERRE}x{LADO_CIERRE}")
plt.subplot(224)
imshow(mascara, title=f"Apertura {LADO_APERTURA}x{LADO_APERTURA}")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "04_mascara.png", dpi=150, bbox_inches="tight")
plt.show()

# --- A. Monedas y dados ---------------------------------------------------------
objetos = []
for contorno in contornos:
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    x, y, w, h = cv2.boundingRect(contorno)
    objetos.append({
        "contorno": contorno,
        "area": area,
        "rho": 4 * np.pi * area / perimetro**2,
        "caja": (x, y, w, h),
        "centro": (x + w // 2, y + h // 2),
    })
# Se numeran de arriba hacia abajo.
objetos.sort(key=lambda objeto: objeto["centro"][1])

print("\nObjeto  Área (px)  Rho    Clase")
monedas = []
dados = []
for numero, objeto in enumerate(objetos, start=1):
    objeto["numero"] = numero
    if objeto["rho"] < RHO_UMBRAL:
        dados.append(objeto)
        clase = "dado"
    else:
        monedas.append(objeto)
        clase = "moneda"
    print(f"{numero:6d}  {objeto['area']:9.0f}  {objeto['rho']:.3f}  {clase}")
print(f"Monedas: {len(monedas)}, dados: {len(dados)}")

img_clases = img_rgb.copy()
for objeto in objetos:
    color = ROJO if objeto["rho"] < RHO_UMBRAL else AZUL
    cv2.drawContours(img_clases, [objeto["contorno"]], -1, color, 14)
    x, y = objeto["centro"]
    texto = f"{objeto['numero']}: {objeto['rho']:.2f}"
    cv2.putText(img_clases, texto, (x - 140, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 0), 7)

plt.figure(figsize=(12, 8))
imshow(img_clases, title="Monedas (azul) y dados (rojo), con su número y factor de forma", color_img=True)
plt.savefig(CARPETA_SALIDA / "05_monedas_y_dados.png", dpi=150, bbox_inches="tight")
plt.show()

# --- B. Tipos de moneda ------------------------------------------------------------
monedas.sort(key=lambda moneda: moneda["area"])
print("\nMoneda  Área (px)  Área / anterior")
tipo = 0
monedas[0]["tipo"] = tipo
print(f"{monedas[0]['numero']:6d}  {monedas[0]['area']:9.0f}")
for anterior, moneda in zip(monedas, monedas[1:]):
    salto = moneda["area"] / anterior["area"]
    if salto > SALTO_AREA:
        tipo += 1
    moneda["tipo"] = tipo
    print(f"{moneda['numero']:6d}  {moneda['area']:9.0f}  {salto:.3f}")

print("\nConteo de monedas:")
for tipo, nombre in enumerate(NOMBRES):
    areas = [moneda["area"] for moneda in monedas if moneda["tipo"] == tipo]
    print(f"  {nombre}: {len(areas)} (áreas de {min(areas):.0f} a {max(areas):.0f} px)")

plt.figure(figsize=(10, 4))
for tipo, nombre in enumerate(NOMBRES):
    posiciones = [i for i, moneda in enumerate(monedas) if moneda["tipo"] == tipo]
    areas = [monedas[i]["area"] for i in posiciones]
    plt.bar(posiciones, areas, color=np.array(COLORES_TIPO[tipo]) / 255, label=nombre)
plt.xticks(range(len(monedas)), [moneda["numero"] for moneda in monedas])
plt.xlabel("Número de moneda (ordenadas por área)")
plt.ylabel("Área (píxeles)")
plt.legend()
plt.title("Área de cada moneda")
plt.savefig(CARPETA_SALIDA / "06_areas_monedas.png", dpi=150, bbox_inches="tight")
plt.show()

# --- C. Valor de los dados ---------------------------------------------------------
# Los puntos son huecos oscuros dentro de la cara blanca. Los de las caras laterales
# se ven de costado y ocupan menos área que los de la cara superior.
plt.figure(figsize=(12, 4 * len(dados)))
for fila, dado in enumerate(dados):
    x, y, w, h = dado["caja"]
    recorte = gris[y:y + h, x:x + w]
    umbral_dado, cara_blanca = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Solo interesa lo que está dentro del dado: el tapete claro también supera el umbral.
    cara_blanca = cv2.bitwise_and(cara_blanca, mascara[y:y + h, x:x + w])
    # Con RETR_CCOMP, los contornos que tienen padre son huecos.
    contornos_dado, jerarquia = cv2.findContours(cara_blanca, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    huecos = [c for c, datos in zip(contornos_dado, jerarquia[0]) if datos[3] != -1]
    fracciones = sorted((cv2.contourArea(c) / dado["area"] for c in huecos), reverse=True)
    puntos = [c for c in huecos if cv2.contourArea(c) / dado["area"] > AREA_PUNTO]
    dado["valor"] = len(puntos)

    print(f"\nDado {dado['numero']}: umbral de Otsu {umbral_dado:.0f}, {len(huecos)} huecos")
    print("  Los 10 huecos más grandes, en % del área del dado:",
          ", ".join(f"{100 * f:.2f}" for f in fracciones[:10]))
    print(f"  Valor: {dado['valor']}")

    img_huecos = cv2.cvtColor(recorte, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_huecos, huecos, -1, ROJO, 3)
    cv2.drawContours(img_huecos, puntos, -1, VERDE, 5)
    plt.subplot(len(dados), 3, 3 * fila + 1)
    imshow(img_rgb[y:y + h, x:x + w], title=f"Dado {dado['numero']}", color_img=True)
    plt.subplot(len(dados), 3, 3 * fila + 2)
    imshow(cara_blanca, title=f"Otsu = {umbral_dado:.0f}")
    plt.subplot(len(dados), 3, 3 * fila + 3)
    imshow(img_huecos, title=f"Puntos (verde): {dado['valor']}", color_img=True)
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "07_dados.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Resultado ------------------------------------------------------------------
img_resultado = img_rgb.copy()
for moneda in monedas:
    cv2.drawContours(img_resultado, [moneda["contorno"]], -1, COLORES_TIPO[moneda["tipo"]], 16)
for dado in dados:
    cv2.drawContours(img_resultado, [dado["contorno"]], -1, ROJO, 16)
    x, y, w, h = dado["caja"]
    cv2.putText(img_resultado, f"Dado: {dado['valor']}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, ROJO, 9)

leyenda = []
for tipo, nombre in enumerate(NOMBRES):
    cantidad = sum(1 for moneda in monedas if moneda["tipo"] == tipo)
    leyenda.append(Patch(color=np.array(COLORES_TIPO[tipo]) / 255, label=f"{nombre}: {cantidad}"))
leyenda.append(Patch(color=np.array(ROJO) / 255, label=f"Dados: {len(dados)}"))

plt.figure(figsize=(12, 8))
imshow(img_resultado, color_img=True)
plt.legend(handles=leyenda, loc="lower left", fontsize=11)
valores = ", ".join(str(dado["valor"]) for dado in dados)
plt.title(f"{len(monedas)} monedas y {len(dados)} dados (valores: {valores})")
plt.savefig(CARPETA_SALIDA / "08_resultado.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nResultado: {len(monedas)} monedas y {len(dados)} dados con valores {valores}")
print(f"Imágenes guardadas en: {CARPETA_SALIDA}")
