# Autodestilación con Grounded DINO
La autodestilación utiliza un modelo base en nuestro caso [Grounded DINO](https://arxiv.org/pdf/2303.05499.pdf) y un modelo target. La idea general es omitir el proceso de etiquetado manual, dejándole esta tarea al modelo base, el cual generará las anotaciones que necesita el modelo target para entrenarse. 

![Autodistill](https://i.imgur.com/qqmdE2B.jpg)

El modelo base es un modelo altamente preciso, pero pesado para utilizarse en producción para tiempo real, el modelo target por su parte es ligero y rápido, acto para dispositivos de borde y que funcionen en tiempo real. Algunos posibles modelo target son las versiones de YOLO y DETR. 

## Grounded DINO
El modelo [DINO](https://arxiv.org/pdf/2203.03605.pdf) (DETR with Improved deNoising anchOr boxes) es un detetector extremo a extremo de ultima generación basado en transformer. DINO mejora el rendimiento y la eficiencia de modelos anteriores de tipo DETR mediante el uso de un método contrastivo para el entrenamiento por denoising, un método mixto de selección de consultas para la inicialización de anclajes y un esquema de doble búsqueda para la predicción de boxes.

[Grounded DINO](https://arxiv.org/pdf/2303.05499.pdf) es una extensión del modelo DINO que se utiliza para la **detección de objetos de conjunto abierto** en la visión por computadora. Grounded DINO utiliza una combinación de técnicas de procesamiento de lenguaje natural y visión por computadora para mejorar la precisión de la detección de objetos y la comprensión de expresiones de referencia.

Aplica una arquitectura de doble codificador y decodificador simple, lo que significa que tiene codificadores separados para las entradas de imagen y texto y un único decodificador que genera la salida final.

![Arquitectura](https://i.imgur.com/REICgNm.png)

## Instalación
- 1- Clonar el repositorio https://github.com/amulet1989/autolabeling
- 2- Generar un .venv y activarlo usando

```
python -m venv .venv
.\.venv\Scripts\activate
```
- 3- instalar las dependencias
```
pip install -r requirements.txt
```
- 4- asegurarse de tener una carpeta "videos" con subcarpetas con los videos de cada producto. Cada carpeta de producto contendrá tres videos de dos minutos más un archivo *ontology.json* con la ontología necesaria definiendo que queremos que el modelo detecte. La onología contiene el diccionario ontológico que tiene el formato {caption0: class0, captio1:class1...} donde *caption* es la indicación enviada al modelo base, y *class* es la etiqueta que se guardará para ese caption en las anotaciones generadas

Ejemplo de archivo ontology.json:
```
{
  "product held by": "7790040133488"
}
```
- 5- Correr el pipeline dentro del script *autolabeling.py* con los argumetos que estimemos convenientes. Para una prueba inicial vamos a elegir un frame_rate alto para obtener pocas imagenes por video:

```
python autolabeling.py --frame_rate 350

```
Finalmente se generará la carpeta *"dataset"* con subcarpetas (un dataset para cada producto) y la carpeta Merged_Dataset con un dataset de todos los objetos individuales listo para entrenar un modelo YOLO o DETR. 