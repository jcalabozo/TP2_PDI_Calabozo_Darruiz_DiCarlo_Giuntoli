"""Problema 2: detección de la placa patente y segmentación de sus caracteres."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
CARPETA_IMAGENES = Path("datos") / "Patentes"
CARPETA_SALIDA = Path("resultados") / "problema2"
IMAGEN_DETALLE = "img01"  # imagen con la que se muestra cada paso en detalle

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
SEPARACION_MAX = 1.5  # espacio máximo entre dos caracteres seguidos

# Márgenes de la placa alrededor de los caracteres, en altos de carácter.
MARGEN_LADOS = 0.4
MARGEN_ARRIBA = 0.8
MARGEN_ABAJO = 0.5

# Para mostrar qué pasa por debajo y por encima de los umbrales elegibles.
SALTO_UMBRAL = 20


def imshow(img, new_fig=False, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """Muestra una imagen con matplotlib (función imshow de la Unidad 3, con los cambios del TP1)."""
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img, interpolation="nearest")
    elif img.dtype == np.uint8:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def binarizar(gris, umbral):
    """Imagen binaria con blanco donde el gris supera el umbral."""
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return binaria


def candidatos(binaria):
    """Cajas (x, y, ancho, alto) de las componentes con forma de carácter, de izquierda a derecha.

    También devuelve cuántas componentes pasan cada filtro, para mostrarlo.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria)
    conteo = {"componentes": n - 1, "alto": 0, "relacion": 0, "area": 0}
    cajas = []
    # La fila 0 de stats es el fondo; cada fila es [x, y, ancho, alto, área].
    for x, y, ancho, alto, area in stats[1:n]:
        if not ALTO_MIN <= alto <= ALTO_MAX:
            continue
        conteo["alto"] += 1
        if not RELACION_MIN <= alto / ancho <= RELACION_MAX:
            continue
        conteo["relacion"] += 1
        if area < AREA_MIN:
            continue
        conteo["area"] += 1
        cajas.append((int(x), int(y), int(ancho), int(alto)))
    # Ordenar tuplas las ordena por su primer valor: la x, de izquierda a derecha.
    return sorted(cajas), conteo


def cadena_mas_larga(cajas):
    """Secuencia más larga de cajas de alto parecido, alineadas y cercanas entre sí.

    Prueba cada caja como primer carácter y recorre las siguientes hacia la
    derecha, agregando las que cumplen las tres condiciones respecto de la
    última agregada.
    """
    mejor = []
    for i, primera in enumerate(cajas):
        alto = primera[3]
        cadena = [primera]
        for caja in cajas[i + 1:]:
            ultima = cadena[-1]
            espacio = caja[0] - (ultima[0] + ultima[2])
            # Cerca: empieza después de la mitad de la anterior y no está lejos.
            cercana = -ultima[2] / 2 < espacio <= SEPARACION_MAX * alto
            # Alto parecido al del primer carácter.
            parecida = abs(caja[3] - alto) <= TOLERANCIA_ALTO * alto
            # Alineada: centro vertical parecido al de la anterior.
            centro_caja = caja[1] + caja[3] / 2
            centro_ultima = ultima[1] + ultima[3] / 2
            alineada = abs(centro_caja - centro_ultima) <= TOLERANCIA_CENTRO * alto
            if cercana and parecida and alineada:
                cadena.append(caja)
        if len(cadena) > len(mejor):
            mejor = cadena
    return mejor


def separaciones(cadena):
    """Espacio horizontal entre cada carácter de la cadena y el siguiente."""
    espacios = []
    for i in range(1, len(cadena)):
        anterior = cadena[i - 1]
        espacios.append(cadena[i][0] - (anterior[0] + anterior[2]))
    return espacios


def es_patente(cadena):
    """Seis caracteres en dos grupos de tres: el espacio más grande está entre el tercero y el cuarto."""
    if len(cadena) != 6:
        return False
    espacios = separaciones(cadena)
    return espacios.index(max(espacios)) == 2


def elegir_umbral(gris):
    """Busca la cadena más larga con cada umbral y elige el umbral del medio entre los que sirven.

    Sirven los umbrales con los que la cadena es una patente válida. Si no hay
    ninguno, se usan los que dan la cadena más larga.
    """
    cadenas = {}
    for umbral in UMBRALES:
        cajas, _ = candidatos(binarizar(gris, umbral))
        cadenas[umbral] = cadena_mas_larga(cajas)
    validos = [umbral for umbral in UMBRALES if es_patente(cadenas[umbral])]
    if validos:
        elegibles = validos
    else:
        largo_maximo = max(len(cadena) for cadena in cadenas.values())
        elegibles = [umbral for umbral in UMBRALES if len(cadenas[umbral]) == largo_maximo]
    umbral = elegibles[len(elegibles) // 2]
    return umbral, cadenas, validos, elegibles


def rectangulo(cajas):
    """Rectángulo (izquierda, arriba, derecha, abajo) que contiene a todas las cajas."""
    izquierda = min(x for x, y, ancho, alto in cajas)
    arriba = min(y for x, y, ancho, alto in cajas)
    derecha = max(x + ancho for x, y, ancho, alto in cajas)
    abajo = max(y + alto for x, y, ancho, alto in cajas)
    return izquierda, arriba, derecha, abajo


def agrandar(izquierda, arriba, derecha, abajo, alto_caracter, forma):
    """Agranda un rectángulo con los márgenes de la placa y lo devuelve como (x, y, ancho, alto)."""
    alto_img, ancho_img = forma[:2]
    x0 = max(int(izquierda - MARGEN_LADOS * alto_caracter), 0)
    x1 = min(int(derecha + MARGEN_LADOS * alto_caracter), ancho_img)
    y0 = max(int(arriba - MARGEN_ARRIBA * alto_caracter), 0)
    y1 = min(int(abajo + MARGEN_ABAJO * alto_caracter), alto_img)
    return x0, y0, x1 - x0, y1 - y0


def enderezar_placa(img_rgb, caracteres, placa, alto_caracter):
    """Rota la imagen para que la placa quede horizontal y recorta la placa.

    Devuelve el recorte, el ángulo y la recta de las bases de los caracteres.
    """
    # La inclinación sale de la recta que mejor ajusta las bases de los caracteres.
    centros_x = [x + ancho / 2 for x, y, ancho, alto in caracteres]
    bases_y = [y + alto for x, y, ancho, alto in caracteres]
    pendiente, ordenada = np.polyfit(centros_x, bases_y, 1)
    angulo = np.degrees(np.arctan(pendiente))

    # Unidad 4: rotación alrededor del centro de la placa.
    x, y, ancho, alto = placa
    rotacion = cv2.getRotationMatrix2D((x + ancho / 2, y + alto / 2), angulo, 1)
    alto_img, ancho_img = img_rgb.shape[:2]
    img_rotada = cv2.warpAffine(img_rgb, rotacion, (ancho_img, alto_img), flags=cv2.INTER_CUBIC)

    # Las esquinas de los caracteres se llevan a la imagen rotada con la misma
    # matriz, y la placa se recorta alrededor de ellas con los mismos márgenes.
    esquinas = []
    for cx, cy, c_ancho, c_alto in caracteres:
        esquinas += [[cx, cy], [cx + c_ancho, cy], [cx, cy + c_alto], [cx + c_ancho, cy + c_alto]]
    esquinas = cv2.transform(np.array([esquinas], dtype=np.float32), rotacion)[0]
    rx, ry, r_ancho, r_alto = agrandar(esquinas[:, 0].min(), esquinas[:, 1].min(),
                                       esquinas[:, 0].max(), esquinas[:, 1].max(), alto_caracter, img_rgb.shape)
    return img_rotada[ry:ry + r_alto, rx:rx + r_ancho], angulo, (pendiente, ordenada)


def procesar(ruta):
    """Aplica todos los pasos a una imagen y devuelve sus resultados en un diccionario."""
    img = cv2.imread(str(ruta))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    umbral, cadenas, validos, elegibles = elegir_umbral(gris)
    caracteres = cadenas[umbral]
    binaria = binarizar(gris, umbral)
    cajas, conteo = candidatos(binaria)

    # Para comparar: lo que se obtiene con un único umbral, el de Otsu.
    umbral_otsu, binaria_otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cajas_otsu, _ = candidatos(binaria_otsu)

    alto_caracter = np.median([alto for x, y, ancho, alto in caracteres])
    izquierda, arriba, derecha, abajo = rectangulo(caracteres)
    placa = agrandar(izquierda, arriba, derecha, abajo, alto_caracter, gris.shape)
    placa_enderezada, angulo, recta = enderezar_placa(img_rgb, caracteres, placa, alto_caracter)

    return {
        "nombre": ruta.stem,
        "img": img_rgb,
        "gris": gris,
        "cadenas": cadenas,
        "validos": validos,
        "elegibles": elegibles,
        "umbral": umbral,
        "binaria": binaria,
        "candidatos": cajas,
        "conteo": conteo,
        "caracteres": caracteres,
        "umbral_otsu": umbral_otsu,
        "cadena_otsu": cadena_mas_larga(cajas_otsu),
        "alto_caracter": alto_caracter,
        "placa": placa,
        "placa_enderezada": placa_enderezada,
        "angulo": angulo,
        "recta": recta,
    }


def zona_alrededor(placa, forma):
    """Zona que se muestra en las figuras: media placa a cada lado y una placa arriba y abajo."""
    x, y, ancho, alto = placa
    x0 = max(x - ancho // 2, 0)
    x1 = min(x + ancho + ancho // 2, forma[1])
    y0 = max(y - alto, 0)
    y1 = min(y + 2 * alto, forma[0])
    return (slice(y0, y1), slice(x0, x1)), (x0, y0)


def dibujar_cajas(cajas, origen, color, estilo="-"):
    """Dibuja cajas sobre el panel actual, que muestra un recorte que empieza en origen."""
    x0, y0 = origen
    for x, y, ancho, alto in cajas:
        # matplotlib ubica el centro de cada píxel en coordenadas enteras, por eso el -0,5.
        esquina = (x - x0 - 0.5, y - y0 - 0.5)
        plt.gca().add_patch(Rectangle(esquina, ancho, alto, fill=False, edgecolor=color, linewidth=1.5, linestyle=estilo))


def dibujar_recta_bases(resultado, origen):
    """Dibuja la recta de las bases de los caracteres sobre el panel actual."""
    pendiente, ordenada = resultado["recta"]
    izquierda, _, derecha, _ = rectangulo(resultado["caracteres"])
    xs = np.array([izquierda, derecha])
    plt.plot(xs - origen[0], pendiente * xs + ordenada - origen[1], color="red", linewidth=1.5)


def curva_de_umbrales(resultado):
    """Grafica, en el panel actual, el largo de la cadena más larga según el umbral."""
    largos = [len(resultado["cadenas"][umbral]) for umbral in UMBRALES]
    plt.plot(UMBRALES, largos, "o-", color="gray", markersize=3)
    plt.plot(resultado["validos"], [6] * len(resultado["validos"]), "o", color="green",
             label="6 caracteres en dos grupos de 3")
    plt.axvline(resultado["umbral"], color="red", label=f"Umbral elegido: {resultado['umbral']}")
    plt.axvline(resultado["umbral_otsu"], color="black", linestyle="--", label=f"Otsu: {resultado['umbral_otsu']:.0f}")
    plt.ylim(-0.3, 7)


CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# --- Detalle del procesamiento de img01 ---------------------------------------------
detalle = procesar(CARPETA_IMAGENES / f"{IMAGEN_DETALLE}.png")
umbral = detalle["umbral"]
conteo = detalle["conteo"]
caracteres = detalle["caracteres"]
gris = detalle["gris"]

print(f"=== Detalle de {IMAGEN_DETALLE} ===")
print(f"Tamaño: {gris.shape[1]} x {gris.shape[0]} píxeles")
print(f"\nUmbral elegido: {umbral}")
print(f"Componentes de la imagen binarizada: {conteo['componentes']}")
print(f"  con alto entre {ALTO_MIN} y {ALTO_MAX} píxeles: {conteo['alto']}")
print(f"  y además con alto/ancho entre {RELACION_MIN} y {RELACION_MAX}: {conteo['relacion']}")
print(f"  y además con área de {AREA_MIN} píxeles o más: {conteo['area']} candidatos")

print("\nCandidatos:")
print("   x    y  ancho  alto  alto/ancho  en la cadena")
for x, y, ancho, alto in detalle["candidatos"]:
    en_cadena = "sí" if (x, y, ancho, alto) in caracteres else ""
    print(f"{x:4d} {y:4d} {ancho:6d} {alto:5d} {alto / ancho:11.2f}  {en_cadena}")

primero = caracteres[0]
print(f"\nCadena más larga: {len(caracteres)} caracteres")
print(f"Alto del primero: {primero[3]} px; se aceptan altos de {primero[3] * (1 - TOLERANCIA_ALTO):.1f} "
      f"a {primero[3] * (1 + TOLERANCIA_ALTO):.1f} px,")
print(f"centros a {TOLERANCIA_CENTRO * primero[3]:.1f} px como mucho y espacios de hasta {SEPARACION_MAX * primero[3]:.1f} px")
print("Carácter  alto  centro vertical  espacio con el anterior")
for i, (x, y, ancho, alto) in enumerate(caracteres):
    espacio = "" if i == 0 else str(x - (caracteres[i - 1][0] + caracteres[i - 1][2]))
    print(f"{i + 1:8d} {alto:5d} {y + alto / 2:16.1f}  {espacio}")
print(f"Patente válida (6 caracteres y el espacio mayor entre el 3.° y el 4.°): {es_patente(caracteres)}")

validos = detalle["validos"]
print(f"\nUmbrales que dan una patente válida: {len(validos)}, de {validos[0]} a {validos[-1]}")
print(f"Umbral de Otsu: {detalle['umbral_otsu']:.0f}, cadena más larga con ese umbral: {len(detalle['cadena_otsu'])}")

izquierda, arriba, derecha, abajo = rectangulo(caracteres)
alto_caracter = detalle["alto_caracter"]
print(f"\nRectángulo de los caracteres (x, y, ancho, alto): {(izquierda, arriba, derecha - izquierda, abajo - arriba)}")
print(f"Alto mediano de los caracteres: {alto_caracter:.0f} píxeles")
print(f"Márgenes: {MARGEN_LADOS * alto_caracter:.1f} píxeles a cada lado, {MARGEN_ARRIBA * alto_caracter:.1f} arriba "
      f"y {MARGEN_ABAJO * alto_caracter:.1f} abajo")
print(f"Placa (x, y, ancho, alto): {detalle['placa']}")
print(f"Recta de las bases: pendiente {detalle['recta'][0]:.3f}, inclinación {detalle['angulo']:.1f}°")

zona, origen = zona_alrededor(detalle["placa"], gris.shape)

# Binarización con el umbral elegido.
plt.figure(figsize=(14, 4.5))
plt.subplot(121)
imshow(gris, title=f"{IMAGEN_DETALLE} en grises")
plt.subplot(122)
imshow(detalle["binaria"], title=f"Binarizada con umbral {umbral}: {conteo['componentes']} componentes")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "01_detalle_binarizacion.png", dpi=120, bbox_inches="tight")
plt.show()

# Candidatos y cadena.
plt.figure(figsize=(14, 4.5))
plt.subplot(121)
imshow(detalle["img"], title=f"Candidatos a carácter: {conteo['area']}", color_img=True)
dibujar_cajas(detalle["candidatos"], (0, 0), "gold")
plt.subplot(122)
imshow(detalle["binaria"][zona], title="Zona de la patente: candidatos (amarillo) y cadena (verde)")
dibujar_cajas(detalle["candidatos"], origen, "gold")
dibujar_cajas(caracteres, origen, "lime")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "02_detalle_candidatos.png", dpi=120, bbox_inches="tight")
plt.show()

# Largo de la cadena según el umbral, y la binarización por debajo, en y por encima
# de los umbrales elegibles.
umbral_bajo = max(detalle["elegibles"][0] - SALTO_UMBRAL, UMBRALES[0])
umbral_alto = min(detalle["elegibles"][-1] + SALTO_UMBRAL, UMBRALES[-1])
plt.figure(figsize=(13, 8))
plt.subplot(2, 1, 1)
curva_de_umbrales(detalle)
plt.xlabel("Umbral")
plt.ylabel("Caracteres en la cadena más larga")
plt.legend(fontsize=9)
plt.title("Largo de la cadena según el umbral")
for i, valor in enumerate([umbral_bajo, umbral, umbral_alto]):
    plt.subplot(2, 3, 4 + i)
    imshow(binarizar(gris, valor)[zona], title=f"Umbral {valor}: cadena de {len(detalle['cadenas'][valor])}")
    dibujar_cajas(detalle["cadenas"][valor], origen, "lime")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "03_detalle_umbrales.png", dpi=120, bbox_inches="tight")
plt.show()

# Placa: rectángulo de los caracteres y márgenes, recta de las bases y placa enderezada.
x, y, ancho, alto = detalle["placa"]
plt.figure(figsize=(14, 4))
plt.subplot(131)
imshow(detalle["img"][zona], title="Caracteres (azul) y placa (rojo)", color_img=True)
dibujar_cajas([(izquierda, arriba, derecha - izquierda, abajo - arriba)], origen, "deepskyblue", "--")
dibujar_cajas([detalle["placa"]], origen, "red")
plt.subplot(132)
imshow(detalle["img"][y:y + alto, x:x + ancho], title="Recta de las bases (rojo)", color_img=True)
dibujar_cajas(caracteres, (x, y), "lime")
dibujar_recta_bases(detalle, (x, y))
plt.subplot(133)
imshow(detalle["placa_enderezada"], title=f"Placa enderezada (rotada {detalle['angulo']:.1f}°)", color_img=True)
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "04_detalle_placa.png", dpi=120, bbox_inches="tight")
plt.show()

# --- Procesamiento de las doce imágenes ----------------------------------------------
print("\n=== Las doce imágenes ===")
resultados = []
for ruta in sorted(CARPETA_IMAGENES.glob("img*.png")):
    resultado = procesar(ruta)
    resultados.append(resultado)
    validos = resultado["validos"]
    if validos:
        texto_validos = f"{len(validos)} umbrales válidos, de {validos[0]} a {validos[-1]}"
    else:
        texto_validos = "ningún umbral válido"
    texto_otsu = "patente válida" if es_patente(resultado["cadena_otsu"]) else "no válida"
    print(f"\n{resultado['nombre']}: {texto_validos}; umbral elegido {resultado['umbral']}")
    print(f"  Otsu: umbral {resultado['umbral_otsu']:.0f}, cadena de {len(resultado['cadena_otsu'])} ({texto_otsu})")
    print(f"  Caracteres: {len(resultado['caracteres'])}, alto mediano {resultado['alto_caracter']:.0f} px, "
          f"espacios (px): {separaciones(resultado['caracteres'])}")
    print(f"  Placa (x, y, ancho, alto): {resultado['placa']}, inclinación {resultado['angulo']:.1f}°")

# Largo de la cadena según el umbral, en las doce imágenes.
plt.figure(figsize=(16, 9))
for i, resultado in enumerate(resultados, start=1):
    plt.subplot(3, 4, i)
    curva_de_umbrales(resultado)
    plt.title(resultado["nombre"])
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "05_largo_segun_umbral.png", dpi=120, bbox_inches="tight")
plt.show()

# Candidatos y cadena alrededor de la patente, con el umbral elegido.
plt.figure(figsize=(16, 9))
for i, resultado in enumerate(resultados, start=1):
    zona, origen = zona_alrededor(resultado["placa"], resultado["gris"].shape)
    plt.subplot(3, 4, i)
    imshow(resultado["binaria"][zona], title=f"{resultado['nombre']}: umbral {resultado['umbral']}")
    dibujar_cajas(resultado["candidatos"], origen, "gold")
    dibujar_cajas(resultado["caracteres"], origen, "lime")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "06_candidatos_y_cadena.png", dpi=120, bbox_inches="tight")
plt.show()

# Placa detectada en cada imagen (ítem A).
plt.figure(figsize=(16, 9))
for i, resultado in enumerate(resultados, start=1):
    plt.subplot(3, 4, i)
    imshow(resultado["img"], title=resultado["nombre"], color_img=True)
    dibujar_cajas([resultado["placa"]], (0, 0), "red")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "07_placas_detectadas.png", dpi=120, bbox_inches="tight")
plt.show()

# Caracteres segmentados en cada placa (ítem B), con la recta de las bases.
plt.figure(figsize=(14, 8))
for i, resultado in enumerate(resultados, start=1):
    x, y, ancho, alto = resultado["placa"]
    plt.subplot(3, 4, i)
    imshow(resultado["img"][y:y + alto, x:x + ancho], color_img=True,
           title=f"{resultado['nombre']}: {len(resultado['caracteres'])} caracteres")
    dibujar_cajas(resultado["caracteres"], (x, y), "lime")
    dibujar_recta_bases(resultado, (x, y))
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "08_caracteres.png", dpi=120, bbox_inches="tight")
plt.show()

# Placas enderezadas.
plt.figure(figsize=(14, 7))
for i, resultado in enumerate(resultados, start=1):
    plt.subplot(3, 4, i)
    imshow(resultado["placa_enderezada"], color_img=True,
           title=f"{resultado['nombre']}: rotada {resultado['angulo']:.1f}°")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "09_placas_enderezadas.png", dpi=120, bbox_inches="tight")
plt.show()

# --- Resumen ------------------------------------------------------------------------
print("\nImagen  Umbral  Caracteres  Inclinación  Placa (x, y, ancho, alto)")
for resultado in resultados:
    print(f"{resultado['nombre']:6s}  {resultado['umbral']:6d}  {len(resultado['caracteres']):10d}"
          f"  {resultado['angulo']:10.1f}°  {resultado['placa']}")
completas = sum(1 for resultado in resultados if len(resultado["caracteres"]) == 6)
con_otsu = sum(1 for resultado in resultados if es_patente(resultado["cadena_otsu"]))
print(f"Placas con los 6 caracteres segmentados: {completas} de {len(resultados)}")
print(f"Con un único umbral (Otsu) habrían sido: {con_otsu} de {len(resultados)}")
print(f"Imágenes guardadas en: {CARPETA_SALIDA}")
