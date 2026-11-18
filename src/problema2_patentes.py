"""Problema 2: detección de la placa patente y segmentación de sus caracteres."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
CARPETA_IMAGENES = Path("datos") / "Patentes"
CARPETA_SALIDA = Path("resultados") / "problema2"

UMBRALES = list(range(40, 231, 5))  # umbrales de gris que se prueban en cada imagen

# Forma de un carácter.
ALTO_MIN = 8  # en píxeles
ALTO_MAX = 40
RELACION_MIN = 1.5  # alto / ancho, según la pista de la cátedra
RELACION_MAX = 3.0
AREA_MIN = 15  # en píxeles

# Cadena de caracteres, en fracciones del alto del primer carácter.
TOLERANCIA_ALTO = 0.3  # diferencia de alto permitida
TOLERANCIA_CENTRO = 0.4  # diferencia permitida entre los centros verticales
SEPARACION_MAX = 1.5  # separación máxima entre dos caracteres seguidos

# Márgenes de la placa alrededor de los caracteres, en altos de carácter.
MARGEN_LADOS = 0.4
MARGEN_ARRIBA = 0.8
MARGEN_ABAJO = 0.5


def imshow(img, title=None, color_img=False, ticks=False):
    """Muestra una imagen en la figura actual, como la función imshow de la Unidad 3."""
    if color_img:
        plt.imshow(img, interpolation="nearest")
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])


def candidatos(binaria):
    """Cajas (x, y, ancho, alto) de las componentes con forma de carácter, de izquierda a derecha."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria)
    cajas = []
    for x, y, ancho, alto, area in stats[1:n]:
        if ALTO_MIN <= alto <= ALTO_MAX and RELACION_MIN <= alto / ancho <= RELACION_MAX and area >= AREA_MIN:
            cajas.append((int(x), int(y), int(ancho), int(alto)))
    return sorted(cajas)


def cadena_mas_larga(cajas):
    """Secuencia más larga de cajas de alto parecido, alineadas y cercanas entre sí."""
    mejor = []
    for i, primera in enumerate(cajas):
        alto = primera[3]
        cadena = [primera]
        for caja in cajas[i + 1:]:
            ultima = cadena[-1]
            separacion = caja[0] - (ultima[0] + ultima[2])
            # La caja tiene que empezar después de la mitad de la anterior y no estar lejos.
            cercana = -ultima[2] / 2 < separacion <= SEPARACION_MAX * alto
            parecida = abs(caja[3] - alto) <= TOLERANCIA_ALTO * alto
            centro_caja = caja[1] + caja[3] / 2
            centro_ultima = ultima[1] + ultima[3] / 2
            alineada = abs(centro_caja - centro_ultima) <= TOLERANCIA_CENTRO * alto
            if cercana and parecida and alineada:
                cadena.append(caja)
        if len(cadena) > len(mejor):
            mejor = cadena
    return mejor


def separaciones(cadena):
    """Espacio horizontal entre cada carácter y el siguiente."""
    return [siguiente[0] - (caja[0] + caja[2]) for caja, siguiente in zip(cadena, cadena[1:])]


def es_patente(cadena):
    """Seis caracteres en dos grupos de tres: el espacio más grande está entre el tercero y el cuarto."""
    if len(cadena) != 6:
        return False
    espacios = separaciones(cadena)
    return espacios.index(max(espacios)) == 2


def dibujar_cajas(cajas, origen, color):
    """Dibuja cajas sobre el panel actual, que muestra un recorte que empieza en origen."""
    x0, y0 = origen
    for x, y, ancho, alto in cajas:
        plt.gca().add_patch(Rectangle((x - x0 - 0.5, y - y0 - 0.5), ancho, alto, fill=False, edgecolor=color, linewidth=1.5))


CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
resultados = []

for ruta in sorted(CARPETA_IMAGENES.glob("img*.png")):
    nombre = ruta.stem
    img = cv2.imread(str(ruta))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alto_img, ancho_img = gris.shape

    # --- Caracteres: se prueba cada umbral y se busca la cadena más larga -----------
    cadenas = {}
    for umbral in UMBRALES:
        _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
        cadenas[umbral] = cadena_mas_larga(candidatos(binaria))
    largos = [len(cadenas[umbral]) for umbral in UMBRALES]
    validos = [umbral for umbral in UMBRALES if es_patente(cadenas[umbral])]

    # Se elige el umbral del medio entre los válidos, lejos de los extremos donde
    # los caracteres se pegan al marco o se cortan. Si ninguno es válido, se usan
    # los umbrales que dan la cadena más larga.
    if validos:
        elegibles = validos
    else:
        elegibles = [umbral for umbral in UMBRALES if len(cadenas[umbral]) == max(largos)]
    umbral = elegibles[len(elegibles) // 2]
    caracteres = cadenas[umbral]
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)

    # Para comparar: lo que se obtiene con un único umbral, el de Otsu.
    umbral_otsu, binaria_otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cadena_otsu = cadena_mas_larga(candidatos(binaria_otsu))

    # --- Placa: el rectángulo de los caracteres agrandado según su alto --------------
    alto_caracter = np.median([caja[3] for caja in caracteres])
    izquierda = min(caja[0] for caja in caracteres)
    derecha = max(caja[0] + caja[2] for caja in caracteres)
    arriba = min(caja[1] for caja in caracteres)
    abajo = max(caja[1] + caja[3] for caja in caracteres)
    px0 = max(int(izquierda - MARGEN_LADOS * alto_caracter), 0)
    px1 = min(int(derecha + MARGEN_LADOS * alto_caracter), ancho_img)
    py0 = max(int(arriba - MARGEN_ARRIBA * alto_caracter), 0)
    py1 = min(int(abajo + MARGEN_ABAJO * alto_caracter), alto_img)
    placa = (px0, py0, px1 - px0, py1 - py0)

    # --- Placa enderezada ---------------------------------------------------------------
    # La inclinación sale de la recta que mejor ajusta las bases de los caracteres.
    centros_x = [x + ancho / 2 for x, y, ancho, alto in caracteres]
    bases_y = [y + alto for x, y, ancho, alto in caracteres]
    pendiente = np.polyfit(centros_x, bases_y, 1)[0]
    angulo = np.degrees(np.arctan(pendiente))
    # Unidad 4: rotación alrededor del centro de la placa, que la deja horizontal.
    centro_placa = (px0 + placa[2] / 2, py0 + placa[3] / 2)
    rotacion = cv2.getRotationMatrix2D(centro_placa, angulo, 1)
    img_rotada = cv2.warpAffine(img_rgb, rotacion, (ancho_img, alto_img), flags=cv2.INTER_CUBIC)
    # Se ubican las esquinas de los caracteres en la imagen rotada y se recorta la
    # placa con los mismos márgenes que antes.
    esquinas = []
    for x, y, ancho, alto in caracteres:
        esquinas += [[x, y], [x + ancho, y], [x, y + alto], [x + ancho, y + alto]]
    esquinas = cv2.transform(np.array([esquinas], dtype=np.float32), rotacion)[0]
    rx0 = max(int(esquinas[:, 0].min() - MARGEN_LADOS * alto_caracter), 0)
    rx1 = min(int(esquinas[:, 0].max() + MARGEN_LADOS * alto_caracter), ancho_img)
    ry0 = max(int(esquinas[:, 1].min() - MARGEN_ARRIBA * alto_caracter), 0)
    ry1 = min(int(esquinas[:, 1].max() + MARGEN_ABAJO * alto_caracter), alto_img)
    placa_enderezada = img_rotada[ry0:ry1, rx0:rx1]

    print(f"\n{nombre}")
    if validos:
        print(f"  Umbrales con 6 caracteres en dos grupos de 3: {len(validos)}, de {validos[0]} a {validos[-1]}")
    else:
        print(f"  Ningún umbral da 6 caracteres; la cadena más larga tiene {max(largos)}")
    texto_otsu = "patente válida" if es_patente(cadena_otsu) else "no es una patente válida"
    print(f"  Umbral de Otsu: {umbral_otsu:.0f}, cadena más larga: {len(cadena_otsu)} ({texto_otsu})")
    print(f"  Umbral elegido: {umbral}, caracteres: {len(caracteres)}, alto mediano: {alto_caracter:.0f} px")
    print(f"  Espacios entre caracteres (px): {separaciones(caracteres)}")
    print(f"  Placa (x, y, ancho, alto): {placa}")
    print(f"  Inclinación de la placa: {angulo:.1f}°")
    resultados.append({
        "nombre": nombre,
        "img": img_rgb,
        "placa": placa,
        "caracteres": caracteres,
        "umbral": umbral,
        "validos": validos,
        "otsu_valido": es_patente(cadena_otsu),
        "placa_enderezada": placa_enderezada,
        "angulo": angulo,
    })

    # --- Figura de las etapas ---------------------------------------------------------
    # Zona alrededor de la placa, para ver la binarización y los candidatos.
    zx0 = max(px0 - placa[2] // 2, 0)
    zx1 = min(px1 + placa[2] // 2, ancho_img)
    zy0 = max(py0 - placa[3], 0)
    zy1 = min(py1 + placa[3], alto_img)

    img_cajas = img_rgb.copy()
    cv2.rectangle(img_cajas, (px0, py0), (px1, py1), (255, 0, 0), 2)

    # Arriba dos paneles y abajo tres.
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    imshow(img_cajas, title=f"{nombre}: placa detectada", color_img=True)
    plt.subplot(2, 2, 2)
    plt.plot(UMBRALES, largos, "o-", color="gray", markersize=3)
    plt.plot(validos, [6] * len(validos), "o", color="green", label="6 caracteres en dos grupos de 3")
    plt.axvline(umbral, color="red", label=f"Umbral elegido: {umbral}")
    plt.axvline(umbral_otsu, color="black", linestyle="--", label=f"Otsu: {umbral_otsu:.0f}")
    plt.xlabel("Umbral")
    plt.ylabel("Caracteres en la cadena más larga")
    plt.legend(fontsize=8)
    plt.title("Largo de la cadena según el umbral")
    plt.subplot(2, 3, 4)
    imshow(binaria[zy0:zy1, zx0:zx1], title=f"Umbral {umbral}: candidatos (amarillo)\ny cadena (verde)")
    dibujar_cajas(candidatos(binaria), (zx0, zy0), "gold")
    dibujar_cajas(caracteres, (zx0, zy0), "lime")
    plt.subplot(2, 3, 5)
    imshow(img_rgb[py0:py1, px0:px1], title=f"Placa y caracteres segmentados: {len(caracteres)}", color_img=True)
    dibujar_cajas(caracteres, (px0, py0), "lime")
    plt.subplot(2, 3, 6)
    imshow(placa_enderezada, title=f"Placa enderezada (rotada {angulo:.1f}°)", color_img=True)
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / f"{nombre}_etapas.png", dpi=120, bbox_inches="tight")
    plt.show()

# --- Resumen de las doce imágenes ---------------------------------------------------
print("\nImagen  Umbral  Caracteres  Inclinación  Placa (x, y, ancho, alto)")
for resultado in resultados:
    print(f"{resultado['nombre']:6s}  {resultado['umbral']:6d}  {len(resultado['caracteres']):10d}"
          f"  {resultado['angulo']:10.1f}°  {resultado['placa']}")
completas = sum(1 for resultado in resultados if len(resultado["caracteres"]) == 6)
con_otsu = sum(1 for resultado in resultados if resultado["otsu_valido"])
print(f"Placas con los 6 caracteres segmentados: {completas} de {len(resultados)}")
print(f"Con un único umbral (Otsu) habrían sido: {con_otsu} de {len(resultados)}")

plt.figure(figsize=(14, 8))
for i, resultado in enumerate(resultados, start=1):
    x, y, ancho, alto = resultado["placa"]
    plt.subplot(3, 4, i)
    imshow(resultado["img"][y:y + alto, x:x + ancho], color_img=True)
    dibujar_cajas(resultado["caracteres"], (x, y), "lime")
    plt.title(f"{resultado['nombre']}: {len(resultado['caracteres'])} caracteres")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "resumen_placas.png", dpi=120, bbox_inches="tight")
plt.show()

plt.figure(figsize=(14, 7))
for i, resultado in enumerate(resultados, start=1):
    plt.subplot(3, 4, i)
    imshow(resultado["placa_enderezada"], color_img=True)
    plt.title(f"{resultado['nombre']}: rotada {resultado['angulo']:.1f}°")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "resumen_placas_enderezadas.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Imágenes guardadas en: {CARPETA_SALIDA}")
