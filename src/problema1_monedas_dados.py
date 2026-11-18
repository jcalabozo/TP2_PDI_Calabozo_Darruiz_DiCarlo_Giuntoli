"""Problema 1: segmentación, clasificación y conteo de monedas y dados."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
IMAGEN = Path("datos") / "monedas.jpg"
CARPETA_SALIDA = Path("resultados") / "problema1"

# Parámetros de la segmentación. El informe explica cómo se eligió cada uno.
SIGMA = 2  # desvío del suavizado gaussiano antes de Canny
CANNY_BAJO = 20  # umbrales de histéresis de Canny
CANNY_ALTO = 60
LARGO_MINIMO = 50  # en píxeles; los tramos de borde más cortos son textura del tapete
LADO_CIERRE = 35  # elemento estructural que cierra los cortes de los contornos
LADO_APERTURA = 41  # elemento estructural que limpia la máscara

# Parámetros de la clasificación.
RHO_UMBRAL = 0.8  # factor de forma: por debajo, el objeto es un dado
SALTO_AREA = 1.10  # una moneda un 10 % más grande que la anterior empieza otro tipo
NOMBRES = ["10 centavos", "1 peso", "50 centavos"]  # tipos de moneda, de menor a mayor
AREA_PUNTO = 0.02  # área mínima de un punto del dado, como fracción del área del dado

# Zona de la moneda 2, la dorada grande de arriba, para ver la morfología en detalle.
ZONA_DETALLE = (slice(362, 762), slice(1969, 2369))

# Colores en RGB para dibujar sobre la imagen.
AZUL = (30, 90, 255)
ROJO = (230, 30, 30)
VERDE = (0, 200, 0)
COLORES_TIPO = [(0, 190, 255), (255, 0, 200), (255, 150, 0)]


def imshow(img, new_fig=False, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """Muestra una imagen con matplotlib (función imshow de la Unidad 3).

    Tiene los mismos cambios que en el TP1: por defecto dibuja en la figura
    actual, porque las figuras tienen varios paneles, y las imágenes de 8 bits
    se muestran con el rango completo, de 0 a 255.
    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    elif img.dtype == np.uint8:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def imfillhole(img):
    """Rellena los huecos de una máscara binaria (función imfillhole_v2 de la Unidad 6)."""
    img_flood_fill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # cv2.floodFill pide una máscara 2 píxeles más grande
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)  # Inunda el fondo desde la esquina.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)  # Lo que no se inundó son los huecos.
    return img | img_flood_fill_inv


def contar_regiones(mascara):
    """Cantidad de regiones blancas de una máscara (sin contar el fondo)."""
    n, _ = cv2.connectedComponents(mascara)
    return n - 1


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
# Unidad 6: la apertura con un elemento más grande que los objetos los borra y
# deja una estimación del fondo; restarla deja un fondo uniforme (top-hat). El
# elemento tiene que superar a la moneda más grande, de unos 360 píxeles de
# diámetro. Por su tamaño, esta apertura tarda unos segundos.
elemento_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (401, 401))
fondo_estimado = cv2.morphologyEx(gris, cv2.MORPH_OPEN, elemento_grande)
top_hat = cv2.subtract(gris, fondo_estimado)
umbral_th, mascara_th = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mascara_th = imfillhole(mascara_th)

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
# Como en los ejemplos de la Unidad 3, se suaviza antes de aplicar Canny.
suavizada = cv2.GaussianBlur(gris, (0, 0), SIGMA)
bordes = cv2.Canny(suavizada, CANNY_BAJO, CANNY_ALTO)

# Cada tramo de borde es una componente conectada. stats[i, cv2.CC_STAT_AREA]
# es la cantidad de píxeles del tramo i, es decir, su largo.
n_tramos, etiquetas, stats, _ = cv2.connectedComponentsWithStats(bordes, connectivity=8)
tramos_largos = []
for i in range(1, n_tramos):  # La etiqueta 0 es el fondo.
    if stats[i, cv2.CC_STAT_AREA] >= LARGO_MINIMO:
        tramos_largos.append(i)
# np.isin marca los píxeles cuya etiqueta está en la lista de tramos largos.
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
elemento_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LADO_APERTURA, LADO_APERTURA))

# Sin dilatar, el relleno no completa ninguna moneda: los contornos tienen cortes.
rellena_sin_dilatar = imfillhole(bordes_largos)
dilatada = cv2.dilate(bordes_largos, elemento_cierre)  # Cierra los cortes de los contornos.
rellena = imfillhole(dilatada)  # Rellena el interior de cada contorno cerrado.
erosionada = cv2.erode(rellena, elemento_cierre)  # Devuelve a los objetos su tamaño.
mascara = cv2.morphologyEx(erosionada, cv2.MORPH_OPEN, elemento_apertura)  # Limpia restos chicos.

print(f"Regiones después de la dilatación: {contar_regiones(dilatada)}")
print(f"Regiones después del relleno: {contar_regiones(rellena)}")
print(f"Regiones después de la erosión: {contar_regiones(erosionada)}")
print(f"Regiones después de la apertura: {contar_regiones(mascara)}")

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

# Detalle de una moneda en cada paso, para ver qué hace cada operación.
pasos = [
    (gris, "Gris"),
    (bordes_largos, "Tramos largos de Canny"),
    (rellena_sin_dilatar, "Relleno sin dilatar"),
    (dilatada, "Dilatación"),
    (rellena, "Dilatación y relleno"),
    (mascara, "Erosión y apertura"),
]
plt.figure(figsize=(12, 8))
for i, (paso, titulo) in enumerate(pasos, start=1):
    plt.subplot(2, 3, i)
    imshow(paso[ZONA_DETALLE], title=titulo)
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "05_detalle_morfologia.png", dpi=150, bbox_inches="tight")
plt.show()

# --- A. Monedas y dados ---------------------------------------------------------
# Factor de forma de la pista de la cátedra, multiplicado por 4 pi para que un
# círculo dé 1. Valores ideales, con radio 1 y lado 1:
area_circulo, perimetro_circulo = np.pi, 2 * np.pi
area_cuadrado, perimetro_cuadrado = 1, 4
print(f"\nFactor de forma de un círculo: {4 * np.pi * area_circulo / perimetro_circulo**2:.3f}")
print(f"Factor de forma de un cuadrado: {4 * np.pi * area_cuadrado / perimetro_cuadrado**2:.3f}")

contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
# Se numeran de arriba hacia abajo, según la fila de su centro.
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
plt.savefig(CARPETA_SALIDA / "06_monedas_y_dados.png", dpi=150, bbox_inches="tight")
plt.show()

# Área y factor de forma de cada objeto en un mismo gráfico.
plt.figure(figsize=(9, 5))
for objeto in objetos:
    color = np.array(ROJO if objeto["rho"] < RHO_UMBRAL else AZUL) / 255
    plt.scatter(objeto["rho"], objeto["area"], color=color)
    plt.annotate(str(objeto["numero"]), (objeto["rho"], objeto["area"]), textcoords="offset points", xytext=(5, 3))
plt.axvline(RHO_UMBRAL, color="gray", linestyle="--", label=f"Umbral de rho = {RHO_UMBRAL}")
plt.axvline(np.pi / 4, color="red", linestyle=":", label="Cuadrado ideal (0,785)")
plt.xlabel("Factor de forma (rho)")
plt.ylabel("Área (píxeles)")
plt.legend()
plt.title("Área y factor de forma de cada objeto")
plt.savefig(CARPETA_SALIDA / "07_area_y_factor_de_forma.png", dpi=150, bbox_inches="tight")
plt.show()

# --- B. Tipos de moneda ------------------------------------------------------------
# Se ordenan las monedas de menor a mayor área y se recorren comparando cada una
# con la anterior: si es más de un 10 % mayor, empieza un tipo nuevo.
monedas.sort(key=lambda moneda: moneda["area"])
print("\nMoneda  Área (px)  Área / anterior")
tipo = 0
monedas[0]["tipo"] = tipo
print(f"{monedas[0]['numero']:6d}  {monedas[0]['area']:9.0f}")
for i in range(1, len(monedas)):
    salto = monedas[i]["area"] / monedas[i - 1]["area"]
    if salto > SALTO_AREA:
        tipo += 1
    monedas[i]["tipo"] = tipo
    print(f"{monedas[i]['numero']:6d}  {monedas[i]['area']:9.0f}  {salto:.3f}")

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
plt.savefig(CARPETA_SALIDA / "08_areas_monedas.png", dpi=150, bbox_inches="tight")
plt.show()

# --- C. Valor de los dados ---------------------------------------------------------
# Los puntos son huecos oscuros dentro de la cara blanca del dado. Los de las caras
# laterales se ven de costado y ocupan menos área que los de la cara superior.
plt.figure(figsize=(12, 4 * len(dados)))
for fila, dado in enumerate(dados):
    x, y, w, h = dado["caja"]
    recorte = gris[y:y + h, x:x + w]
    umbral_dado, cara_blanca = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Solo interesa lo que está dentro del dado: en el lado derecho de la imagen,
    # el tapete también supera el umbral.
    cara_blanca = cv2.bitwise_and(cara_blanca, mascara[y:y + h, x:x + w])

    # Con RETR_CCOMP, OpenCV organiza los contornos en dos niveles: los bordes
    # exteriores de las regiones blancas y, dentro de ellos, los bordes de sus
    # huecos. Para cada contorno, jerarquia[0][i] es [siguiente, anterior,
    # primer hijo, padre]: un contorno con padre (distinto de -1) es un hueco.
    contornos_dado, jerarquia = cv2.findContours(cara_blanca, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    huecos = []
    for i, contorno in enumerate(contornos_dado):
        if jerarquia[0][i][3] != -1:
            huecos.append(contorno)

    # Un hueco es un punto de la cara superior si ocupa más del 2 % del dado.
    puntos = []
    fracciones = []
    for hueco in huecos:
        fraccion = cv2.contourArea(hueco) / dado["area"]
        fracciones.append(fraccion)
        if fraccion > AREA_PUNTO:
            puntos.append(hueco)
    dado["valor"] = len(puntos)

    fracciones.sort(reverse=True)
    mayores = ", ".join(f"{100 * fraccion:.2f}" for fraccion in fracciones[:10])
    print(f"\nDado {dado['numero']}: umbral de Otsu {umbral_dado:.0f}, {len(huecos)} huecos")
    print("  Los 10 huecos más grandes, en % del área del dado:")
    print(f"  {mayores}")
    print(f"  Valor: {dado['valor']}")

    img_huecos = cv2.cvtColor(recorte, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_huecos, huecos, -1, ROJO, 3)
    cv2.drawContours(img_huecos, puntos, -1, VERDE, 5)
    plt.subplot(len(dados), 3, 3 * fila + 1)
    imshow(img_rgb[y:y + h, x:x + w], title=f"Dado {dado['numero']}", color_img=True)
    plt.subplot(len(dados), 3, 3 * fila + 2)
    imshow(cara_blanca, title=f"Cara blanca (Otsu = {umbral_dado:.0f})")
    plt.subplot(len(dados), 3, 3 * fila + 3)
    imshow(img_huecos, title=f"Huecos (rojo) y puntos (verde): {dado['valor']}", color_img=True)
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "09_dados.png", dpi=150, bbox_inches="tight")
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

valores = ", ".join(str(dado["valor"]) for dado in dados)
plt.figure(figsize=(12, 8))
imshow(img_resultado, title=f"{len(monedas)} monedas y {len(dados)} dados (valores: {valores})", color_img=True)
plt.legend(handles=leyenda, loc="lower left", fontsize=11)
plt.savefig(CARPETA_SALIDA / "10_resultado.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nResultado: {len(monedas)} monedas y {len(dados)} dados con valores {valores}")
print(f"Imágenes guardadas en: {CARPETA_SALIDA}")
