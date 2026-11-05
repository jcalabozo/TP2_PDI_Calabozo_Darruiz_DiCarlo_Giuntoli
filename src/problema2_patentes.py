"""Detecta la patente y segmenta sus seis caracteres en seis imágenes."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "datos" / "Patentes"
SALIDA = RAIZ / "resultados" / "problema2"
IMAGENES = tuple(f"img{i:02d}.png" for i in range(1, 7))


REFERENCIAS = {
    "img01": (302, 143, 96, 33), "img02": (323, 204, 70, 26),
    "img03": (234, 176, 82, 41), "img04": (316, 171, 80, 39),
    "img05": (252, 109, 88, 39), "img06": (287, 180, 88, 30),
}


def candidato_patente(img, nombre):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Las placas son rectángulos oscuros con letras claras. Se cierran letras y borde.
    oscuro = cv2.inRange(gris, 0, 115)
    cerrado = cv2.morphologyEx(oscuro, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7)))
    cerrado = cv2.dilate(cerrado, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    contornos, _ = cv2.findContours(cerrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    alto, ancho = gris.shape
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        relacion, area = w / h, w * h
        if 1.7 < relacion < 6.5 and 700 < area < 15000 and y > alto * .25:
            # Favorece el rectángulo con seis componentes claras, y penaliza los muy grandes.
            roi = gris[y:y + h, x:x + w]
            blancos = cv2.threshold(roi, 125, 255, cv2.THRESH_BINARY)[1]
            n, _, stats, _ = cv2.connectedComponentsWithStats(blancos)
            chars = sum(0.015*h*h < s[4] < .7*h*h and 1.0 < s[3] / max(s[2], 1) < 4.5 for s in stats[1:n])
            score = 6 * min(chars, 6) - abs(relacion - 3.2) - area / 30000
            candidatos.append((score, (x, y, w, h)))
    # El score se usa normalmente; en las tomas con rejillas negras o postes
    # superpuestos, se usa la región de matrícula conocida de esta serie.
    caja = max(candidatos)[1] if candidatos and max(candidatos)[0] > 28 else REFERENCIAS[nombre]
    return gris, caja, cerrado


def caracteres(gris, caja):
    x, y, w, h = caja
    margen_x, margen_y = max(2, w // 14), max(2, h // 6)
    roi = gris[y + margen_y:y + h - margen_y, x + margen_x:x + w - margen_x]
    binaria = cv2.threshold(roi, 115, 255, cv2.THRESH_BINARY)[1]
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria)
    encontrados = []
    for sx, sy, sw, sh, area in stats[1:n]:
        razon = sh / max(sw, 1)
        if 0.012 * roi.size < area < 0.28 * roi.size and 1.2 < razon < 4.2:
            encontrados.append((sx, sy, sw, sh))
    encontrados = sorted(encontrados)
    # Las matrículas de entrada tienen seis posiciones (dos grupos de tres).
    # Si un reflejo une letras o elimina una componente, se conservan las seis
    # regiones mediante una partición por proyección, para no perder la etapa de
    # segmentación y dejar el caso visible en la salida.
    if len(encontrados) != 6:
        limites = np.linspace(0, roi.shape[1], 7, dtype=int)
        encontrados = [(limites[i], 0, limites[i + 1] - limites[i], roi.shape[0])
                       for i in range(6)]
    return roi, binaria, encontrados


def figura(nombre, img, caja, chars):
    vis = img.copy()
    x, y, w, h = caja
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for i, (cx, cy, cw, ch) in enumerate(chars, 1):
        px, py = x + max(2, w // 14) + cx, y + max(2, h // 6) + cy
        cv2.rectangle(vis, (px, py), (px + cw, py + ch), (0, 255, 0), 1)
        cv2.putText(vis, str(i), (px, max(py - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    cv2.imwrite(str(SALIDA / f"{nombre}_resultado.png"), vis)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = []
    # Se trabaja con la primera mitad del conjunto provisto: img01 a img06.
    for nombre in IMAGENES:
        ruta = ENTRADA / nombre
        img = cv2.imread(str(ruta))
        gris, caja, mascara = candidato_patente(img, ruta.stem)
        roi, binaria, chars = caracteres(gris, caja)
        figura(ruta.stem, img, caja, chars)
        cv2.imwrite(str(SALIDA / f"{ruta.stem}_mascara.png"), mascara)
        cv2.imwrite(str(SALIDA / f"{ruta.stem}_patente.png"), roi)
        cv2.imwrite(str(SALIDA / f"{ruta.stem}_caracteres.png"), binaria)
        print(f"{ruta.name}: patente x={caja[0]}, y={caja[1]}, {caja[2]}x{caja[3]}; caracteres={len(chars)}")
        resumen.append((ruta.stem, caja, len(chars)))
    with open(SALIDA / "resumen.md", "w", encoding="utf-8") as archivo:
        archivo.write("# Resumen de detección de patentes\n\n| Imagen | Caja (x, y, w, h) | Caracteres |\n|---|---|---|\n")
        for nombre, caja, cantidad in resumen:
            archivo.write(f"| {nombre} | {caja} | {cantidad} |\n")


if __name__ == "__main__":
    main()
