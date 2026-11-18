## Problema 1: Detección y clasificación de monedas y dados

### Descripción del problema

La imagen `monedas.jpg` muestra monedas de distinto valor y tamaño y dos dados sobre un tapete de intensidad no uniforme. La consigna pide:

- **A)** Segmentar automáticamente las monedas y los dados.
- **B)** Clasificar los distintos tipos de monedas y contarlas.
- **C)** Determinar el valor de la cara superior de cada dado y contarlos.

La solución está en `src/problema1_monedas_dados.py`. El script muestra cada etapa en pantalla, imprime sus resultados y guarda las figuras en `resultados/problema1/`.

### Análisis de la imagen

Antes de segmentar, el script mide el nivel de gris del tapete en los dos bordes de la imagen, donde no hay objetos, y prueba un umbral global con el método de Otsu (Unidad 3):

```text
Tamaño: 3871 x 2681 píxeles
Gris medio de las 100 columnas de la izquierda: 52
Gris medio de las 100 columnas de la derecha: 171
Umbral de Otsu sobre toda la imagen: 115
```

El tapete pasa de un gris medio de 52 a la izquierda a 171 a la derecha. Por eso el histograma de la Figura 1 tiene dos picos, que no son el fondo y los objetos sino las dos mitades del tapete. Otsu pone el umbral entre ellos, en 115, y la máscara resultante separa la mitad derecha del tapete de la izquierda. Las monedas de la derecha, además, quedan más oscuras que su fondo: en la máscara se ven negras sobre blanco.

![Imagen original, histograma de grises y umbral de Otsu sobre toda la imagen.](../resultados/problema1/01_fondo_y_otsu.png)

Un umbral único no sirve: el mismo nivel de gris es fondo de un lado de la imagen y moneda del otro.

### Intento con top-hat

El apunte de la Unidad 6 propone, para estos casos, obtener un fondo uniforme restándole a la imagen su apertura con un elemento estructural más grande que los objetos: la transformación top-hat. El apunte de la Unidad 3 usa la misma idea como umbral local. Se usó una elipse de 401×401 píxeles, más grande que la moneda más grande: su área, de 103 016 píxeles según la tabla de la sección 1.5, equivale a un círculo de unos 362 píxeles de diámetro. Después, el top-hat se umbraliza con Otsu y se rellenan los huecos:

```text
Top-hat: umbral de Otsu 72, 5763 regiones
Top-hat: la región más grande tiene 1065620 píxeles
```

En esta imagen el top-hat no alcanza (Figura 2):

- **El fondo estimado no es suave.** Donde las monedas están cerca, la elipse no entra entre ellas sin tocar zonas más oscuras, y la apertura queda por debajo del tapete real. El centro del tapete conserva brillo en el top-hat y, al umbralizar, se une con las monedas: la región más grande mide 1 065 620 píxeles, diez veces una moneda grande.
- **El top-hat solo resalta lo que es más claro que el fondo.** Las monedas que son más oscuras que el tapete, como las de la derecha, quedan apenas marcadas por su borde.

![Fondo estimado con una apertura de 401×401, top-hat, y top-hat umbralado con Otsu y relleno de huecos.](../resultados/problema1/02_intento_top_hat.png)

El problema no es solo el gradiente del fondo: hay monedas con un nivel de gris parecido al del tapete que las rodea. Lo que las separa siempre es el borde, así que la segmentación se hizo a partir de los bordes.

### Segmentación por bordes (ítem A)

**Bordes.** Se suaviza la imagen con un filtro gaussiano de desvío 2 y se aplica Canny (Unidad 3) con umbrales 20 y 60. Canny responde a los cambios locales de intensidad, no al nivel absoluto, así que el gradiente lento del tapete no genera bordes. La textura del tapete sí genera muchos tramos cortos. Cada tramo es una componente conectada de la imagen de bordes, y se descartan los de menos de 50 píxeles:

```text
Canny: 3586 tramos de borde, 936 de 50 píxeles o más
```

La Figura 3 muestra el efecto en la zona central, donde el tapete tiene más textura: los contornos de las monedas y los dados se mantienen y desaparecen los tramos sueltos del tapete. Si se dejaran, la dilatación del paso siguiente los engordaría y podrían unir monedas vecinas.

![Zona central de la imagen, bordes de Canny y tramos de 50 píxeles o más.](../resultados/problema1/03_bordes.png)

**Máscara.** Los contornos de Canny tienen cortes, así que no se pueden rellenar directamente. Se aplica la secuencia de la Figura 4, con las herramientas de la Unidad 6:

1. **Dilatación** con una elipse de 35×35, que cierra los cortes de los contornos.
2. **Relleno de huecos** con la función `imfillhole_v2` de la Unidad 6: se inunda el fondo desde una esquina con `cv2.floodFill`, y lo que no se inunda queda dentro de un contorno cerrado.
3. **Erosión** con la misma elipse, que devuelve a los objetos el tamaño que tenían antes de la dilatación.
4. **Apertura** con una elipse de 41×41, que elimina los restos chicos y alisa el borde de los objetos.

```text
Regiones después de la erosión: 23
Regiones después de la apertura: 19
```

La apertura elimina cuatro regiones chicas, y quedan 19 objetos. Los umbrales de Canny y los tamaños de los elementos estructurales se eligieron probando sobre la imagen. Con umbrales más altos o un elemento de cierre más chico, algunos contornos quedan abiertos y el relleno no completa la moneda; con un elemento de cierre más grande, se unen monedas vecinas.

![Dilatación, relleno de huecos, erosión y apertura.](../resultados/problema1/04_mascara.png)

### Monedas y dados (ítem A)

Para separar monedas de dados se usa el factor de forma de la pista de la cátedra, multiplicado por $4\pi$ para que un círculo dé 1:

$$
\rho = \frac{4 \pi A}{P^2}
$$

donde $A$ es el área del contorno externo del objeto (`cv2.contourArea`) y $P$ su perímetro (`cv2.arcLength`). Los objetos se numeran de arriba hacia abajo:

```text
Objeto  Área (px)  Rho    Clase
     1      72020  0.778  dado
     2     103016  0.876  moneda
     3      59954  0.881  moneda
     4      61522  0.868  moneda
     5      87700  0.861  moneda
     6      84432  0.875  moneda
     7      58230  0.896  moneda
     8      62638  0.720  dado
     9     102796  0.866  moneda
    10      60859  0.873  moneda
    11      86090  0.892  moneda
    12      58726  0.850  moneda
    13     101714  0.842  moneda
    14      60166  0.886  moneda
    15      61962  0.859  moneda
    16      60479  0.885  moneda
    17      58434  0.885  moneda
    18      87098  0.832  moneda
    19      87332  0.890  moneda
Monedas: 17, dados: 2
```

Las monedas dan entre 0,832 y 0,896, y los dados, 0,720 y 0,778. Ninguna moneda llega a 1 por dos motivos: el contorno sigue el borde de los píxeles en escalones, así que `cv2.arcLength` mide un perímetro más largo que el de la circunferencia real, y el borde de la máscara tiene pequeñas irregularidades (Figura 4). Con un rango de 0,9 a 1,1 para los círculos, todas las monedas quedarían como no circulares. Por eso el umbral se eligió con los valores medidos: un objeto con $\rho < 0{,}8$ es un dado. Ese valor queda entre el dado de mayor $\rho$ y la moneda de menor $\rho$, y separa los 17 círculos de los 2 cuadrados (Figura 5).

![Monedas en azul y dados en rojo, con el número y el factor de forma de cada objeto.](../resultados/problema1/05_monedas_y_dados.png)

### Tipos de moneda (ítem B)

Las monedas de distinto valor tienen distinto tamaño, así que se clasifican por área. El script las ordena de menor a mayor y calcula cuánto mayor es cada una que la anterior:

```text
Moneda  Área (px)  Área / anterior
     7      58230
    17      58434  1.004
    12      58726  1.005
     3      59954  1.021
    14      60166  1.004
    16      60479  1.005
    10      60859  1.006
     4      61522  1.011
    15      61962  1.007
     6      84432  1.363
    11      86090  1.020
    18      87098  1.012
    19      87332  1.003
     5      87700  1.004
    13     101714  1.160
     9     102796  1.011
     2     103016  1.002
```

Entre monedas consecutivas el área crece como mucho un 2,1 %, salvo en dos saltos: 36,3 % y 16,0 %. Se toma como criterio que una moneda más de un 10 % mayor que la anterior empieza un tipo nuevo, y así quedan tres grupos (Figura 6). El nombre de cada grupo sale de lo que se lee en sus monedas: «10 CENTAVOS» en las monedas 3, 12 y 15, «UN PESO» en las bimetálicas (por ejemplo, la 11) y un «50», gastado, en la moneda 2.

```text
Conteo de monedas:
  10 centavos: 9 (áreas de 58230 a 61962 px)
  1 peso: 5 (áreas de 84432 a 87700 px)
  50 centavos: 3 (áreas de 101714 a 103016 px)
```

![Área de cada moneda, ordenadas de menor a mayor y coloreadas por tipo.](../resultados/problema1/06_areas_monedas.png)

### Valor de los dados (ítem C)

Los puntos de un dado son huecos oscuros dentro de su cara blanca. Para cada dado:

1. Se recorta su rectángulo envolvente y se umbraliza el recorte con Otsu, que separa la cara blanca de lo oscuro.
2. Se conserva solo lo que está dentro de la máscara del dado, porque en el lado derecho de la imagen el tapete también supera el umbral.
3. Se buscan los contornos con `cv2.RETR_CCOMP`, que organiza la jerarquía en dos niveles: los contornos que tienen un contorno padre son huecos.

```text
Dado 1: umbral de Otsu 125, 20 huecos
  Los 10 huecos más grandes, en % del área del dado: 3.14, 3.10, 3.07, 3.07, 3.04, 1.26, 1.20, 1.16, 0.01, 0.01
  Valor: 5

Dado 8: umbral de Otsu 126, 4 huecos
  Los 10 huecos más grandes, en % del área del dado: 3.88, 3.67, 3.61, 0.00
  Valor: 3
```

No todos los huecos son puntos de la cara superior:

- **Puntos de la cara superior:** ocupan entre 3,04 % y 3,88 % del área del dado.
- **Puntos de la cara lateral del dado 1:** entre 1,16 % y 1,26 %. Se ven de costado, achatados, y por eso ocupan menos de la mitad. En el dado 8, las caras laterales quedan oscuras en la binarización, así que sus puntos no forman huecos.
- **El resto:** como mucho 0,01 %. Son reflejos y ruido de la binarización.

Se cuentan como puntos los huecos que ocupan más del 2 % del área del dado. El dado 1 vale 5 y el dado 8 vale 3 (Figura 7).

![Recorte de cada dado, cara blanca umbralizada con Otsu, y huecos: en verde los puntos contados, en rojo el resto.](../resultados/problema1/07_dados.png)

### Resultado y conclusiones

```text
Resultado: 17 monedas y 2 dados con valores 5, 3
```

La Figura 8 resume los tres ítems: 9 monedas de 10 centavos, 5 de 1 peso, 3 de 50 centavos, y dos dados que valen 5 y 3.

![Resultado final: monedas coloreadas por tipo y dados con su valor.](../resultados/problema1/08_resultado.png)

El fondo no uniforme impide umbralar la imagen directamente, y el top-hat no lo resuelve en esta imagen porque las monedas están cerca entre sí y algunas son más oscuras que el tapete. Los bordes, en cambio, no dependen del nivel de gris absoluto, y la morfología los convierte en una máscara con un objeto por moneda o dado. Con esa máscara, el factor de forma separa monedas de dados, y el área separa los tres tipos de moneda con saltos de al menos 16 %, contra un 2,1 % como máximo dentro de cada tipo. En los dados, el área de cada hueco distingue los puntos de la cara superior de los de las caras laterales.

Los umbrales de Canny y los elementos estructurales se ajustaron para esta imagen. Con otra escala o iluminación habría que revisarlos, aunque los criterios de clasificación (factor de forma, saltos de área y área relativa de los puntos) son proporciones y no dependen del tamaño de la imagen.

## Problema 2: Detección y segmentación de patentes

### Detección de la placa

Se umbralizan las zonas oscuras de cada imagen y se aplican cierre y dilatación con elementos rectangulares horizontales. Los contornos candidatos se filtran por área, relación ancho/alto y posición; el puntaje favorece el rectángulo con componentes claras internas y relación de aspecto de patente.

### Segmentación de caracteres

Se elimina el margen de la placa, se binarizan las letras claras y se buscan componentes conectadas. Se retienen las de área intermedia y relación alto/ancho entre 1,2 y 4,2, consistente con la pista de la cátedra (1,5 a 3). Las componentes se ordenan de izquierda a derecha; las máscaras, recortes y cajas se guardan para la primera mitad del conjunto: `img01.png` a `img06.png`.

![Ejemplo de placa detectada y componentes segmentadas, imagen 06.](../resultados/problema2/img06_resultado.png)

### Dificultades y conclusión

Rejillas, paragolpes y sombras introducen rectángulos oscuros parecidos a una matrícula, por lo que se combinan forma, ubicación y estructura interna. Los parámetros se expresan como proporciones del recorte para tolerar cambios de escala. En una colección mayor, convendría rectificar la perspectiva antes de segmentar y validar el patrón de dos grupos de tres caracteres.
