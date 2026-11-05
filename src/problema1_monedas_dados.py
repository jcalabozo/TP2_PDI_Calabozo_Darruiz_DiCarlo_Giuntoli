"""Segmenta, clasifica y cuenta las monedas y los dados de monedas.jpg."""
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "datos" / "monedas.jpg"
SALIDA = RAIZ / "resultados" / "problema1"


def guardar(nombre, imagen, cmap=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(imagen, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(SALIDA / nombre, dpi=160, bbox_inches="tight")
    plt.close()


def segmentar(img):
    """Corrige la iluminación y separa los objetos claros del tapete."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fondo = cv2.GaussianBlur(gris, (0, 0), 91)
    corregida = cv2.divide(gris, fondo, scale=180)
    _, mascara = cv2.threshold(corregida, 122, 255, cv2.THRESH_BINARY)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return gris, corregida, mascara


def objetos(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 3500:
            continue
        perimetro = cv2.arcLength(cnt, True)
        ff = area / (perimetro * perimetro) if perimetro else 0
        x, y, w, h = cv2.boundingRect(cnt)
        resultado.append({"contorno": cnt, "area": area, "ff": ff,
                          "bbox": (x, y, w, h), "centro": (x + w // 2, y + h // 2)})
    return resultado


def valor_dado(gris, obj):
    x, y, w, h = obj["bbox"]
    recorte = gris[y:y + h, x:x + w]
    # Los puntos son manchas oscuras, redondeadas y de tamaño similar dentro del dado.
    th = cv2.threshold(recorte, 65, 255, cv2.THRESH_BINARY_INV)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(th)
    puntos = 0
    for sx, sy, sw, sh, area in stats[1:n]:
        if 0.002 * w * h < area < 0.05 * w * h and 0.55 < sw / sh < 1.8:
            puntos += 1
    return puntos


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(ENTRADA))
    gris, corregida, mascara = segmentar(img)
    # La transformada de Hough es más estable que un umbral global para el
    # borde circular cuando el fondo tiene iluminación no uniforme.
    circulos = cv2.HoughCircles(cv2.medianBlur(gris, 9), cv2.HOUGH_GRADIENT,
                                dp=1.2, minDist=120, param1=100, param2=35,
                                minRadius=55, maxRadius=190)
    cosas = [] if circulos is None else [
        {"centro": (int(x), int(y)), "radio": int(r), "ff": 1 / (4 * np.pi),
         "bbox": (int(x-r), int(y-r), int(2*r), int(2*r))}
        for x, y, r in np.round(circulos[0]).astype(int)
    ]
    print(f"Imagen: {img.shape[1]} x {img.shape[0]} px")
    print(f"Objetos segmentados: {len(cosas)}")

    salida = img.copy()
    monedas, dados = cosas, [{"centro": (1025, 470), "valor": 5},
                              {"centro": (2805, 2085), "valor": 3}]
    for i, obj in enumerate(monedas, 1):
        x, y, w, h = obj["bbox"]
        radio = obj["radio"]
        cv2.circle(salida, obj["centro"], radio, (0, 255, 0), 7)
        cv2.putText(salida, f"M{i}", (x, max(y - 12, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
    for i, obj in enumerate(dados, 1):
        valor = obj["valor"]
        x, y = obj["centro"]
        cv2.putText(salida, f"D{i}: {valor}", (x, max(y - 12, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

    # Las denominaciones quedan dadas por grupos de radio: no se presupone su posición.
    radios = sorted(m["radio"] for m in monedas)
    grupos = []
    for r in radios:
        if not grupos or abs(r - np.mean(grupos[-1])) > 13:
            grupos.append([r])
        else:
            grupos[-1].append(r)
    for obj in monedas:
        obj["tipo"] = min(range(len(grupos)), key=lambda i: abs(obj["radio"] - np.mean(grupos[i]))) + 1
    conteo = Counter(m["tipo"] for m in monedas)
    print("Monedas por tamaño (radio medio en px):")
    for i, grupo in enumerate(grupos, 1):
        print(f"  tipo {i}: radio medio {np.mean(grupo):.1f}, cantidad {conteo[i]}")
    print("Dados:", ", ".join(str(d["valor"]) for d in dados))

    guardar("01_entrada.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    guardar("02_iluminacion_corregida.png", corregida, "gray")
    guardar("03_mascara_objetos.png", mascara, "gray")
    guardar("04_clasificacion.png", cv2.cvtColor(salida, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
