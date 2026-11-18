# Prompt: convertir `facia.py` (ASCII de cámara en consola) en una app de escritorio con UI

## Tu rol y objetivo

Eres un desarrollador senior de Python y visión por computador. Vas a transformar `test-01/facia.py`, un script que muestra la cámara web como arte ASCII en la terminal, en una **aplicación de escritorio con interfaz gráfica**. El objetivo principal es que **la persona frente a la cámara se reconozca lo más posible** en el render: rasgos de la cara legibles, imagen estable y sin parpadeo. También debe seguir funcionando el modo consola.

Trabaja por fases (ver al final). Al terminar cada fase, ejecuta las pruebas y reporta los resultados reales, incluidos los fallos. No digas que algo funciona si no lo has ejecutado.

## Entorno (verificado)

- macOS (Darwin 25), Python **3.15.2**, OpenCV **5.0.0** (`opencv-python`), numpy **2.5.4**, Tk 9.0.
- **PySide6 6.12.0 se puede instalar** (rueda `abi3` universal2). Pillow y psutil **no** están instalados.
- Hay **dos cámaras**: "FaceTime HD" y una Cámara de Continuidad (iPhone). El índice `0` puede ser el iPhone.
- **OpenCV 5 no tiene `cv2.CascadeClassifier` ni los archivos Haar** en `cv2.data`. Para detectar rostros usa `cv2.FaceDetectorYN` con el modelo YuNet (`face_detection_yunet_2023mar.onnx`, de opencv_zoo). No escribas código de Haar cascades de OpenCV 4.
- Crea un entorno virtual `.venv` dentro de `test-01/`. Usa `PySide6-Essentials` (sin WebEngine) y `opencv-python-headless` (desinstala antes `opencv-python` dentro del venv; los dos paquetes chocan).

## Validación del código actual (resultados medidos, no supuestos)

Lo que **está bien** y hay que conservar:
- Memoria estable: +0,7 KB en 2.500 frames y RSS fijo en 91 MB. La cola de tamaño 1 limita los frames en memoria. No hay fugas.
- Comprueba `isOpened()`, `read()` tiene timeout, el hilo termina limpio y el cursor de la terminal se restaura.
- 6,5 ms por frame a 720p y 13 ms a 1080p.

Defectos que debes corregir:
1. **Parpadeo severo**: con una escena estática y ruido de sensor leve (σ=3), cambia el **59–68 % de los caracteres en cada frame**. Esto es lo que más daña la legibilidad. Causas:
   - Paleta de 70 caracteres: cada carácter cubre solo unos 3,6 niveles de gris, así que el ruido los alterna.
   - CLAHE aplicado sobre la imagen ya reducida: con tiles de 8×8, cada tile queda en 15×4 px, lo que amplifica el ruido.
   - `NORM_MINMAX` por frame: depende de píxeles extremos y hace "bombear" el brillo.
   - Sin suavizado temporal.
   Experimento ya medido: con CLAHE a 320 px de ancho antes de reducir, una paleta de 10 caracteres y una EMA temporal de 0,4, el parpadeo baja al **0,6 %**.
2. **Un solo fallo de lectura cierra la app**: `_update` pone `stopped=True` al primer `ret == False`. En macOS, AVFoundation falla algunas lecturas al calentar la cámara y la Cámara de Continuidad se desconecta. Reproducido con una captura simulada.
3. **Cuelgue con terminal de 1 línea**: `rows - 1 = 0` provoca un error en `cv2.resize`.
4. **`cap.release()` se llama desde el hilo principal** mientras el hilo de captura puede estar bloqueado en `read()`. Si `join` agota su timeout, puede romper AVFoundation (riesgo, no reproducido).
5. `INVERT` se evalúa al importar el módulo, así que cambiarlo en tiempo de ejecución no tiene efecto.
6. No se puede elegir cámara ni resolución. `CHAR_ASPECT = 0.5` está fijo en vez de medirse con la fuente real.
7. El array de caracteres `<U1` ocupa 4 bytes por carácter. Usa índices `uint8` y una tabla de bytes.
8. La paleta de 70 caracteres está ordenada para otra fuente, no para Menlo/SF Mono. Su orden de densidad no se ha medido con la fuente real.

## Arquitectura requerida

```
test-01/
  facia.py                 # se mantiene como lanzador de consola (delegando en el paquete)
  facia_app/
    __main__.py            # python -m facia_app  (UI)   |  --terminal  (consola)
    config.py              # @dataclass Settings + persistencia con QSettings
    capture.py             # CameraSource: hilo de captura, estados, reconexión
    engine.py              # funciones PURAS numpy/cv2: frame -> índices de nivel (sin Qt)
    glyphs.py              # atlas de glifos con QPainter, medición de densidad, paletas
    render.py              # índices -> imagen (atlas) / texto / color
    face.py                # YuNet opcional + suavizado de bbox + recorte
    terminal.py            # renderer de consola reutilizando engine
    ui/main_window.py, ui/controls.py, ui/viewport.py
  models/face_detection_yunet_2023mar.onnx   # opcional; si falta, desactivar la función con aviso
  tests/test_engine.py, tests/test_capture.py
  requirements.txt         # versiones fijadas
```

`engine.py` no puede importar Qt. Recibe arrays y devuelve arrays, para poder probarlo sin cámara ni UI.

## Captura de cámara (`capture.py`)

- Usa `cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)` en macOS. Pide 1280×720 a 30 fps, lee los valores reales y muéstralos en la UI. Usa `CAP_PROP_BUFFERSIZE=1` si el backend lo admite.
- Guarda solo el último frame en un slot protegido con `threading.Lock`, junto con un contador `seq`. No uses `Queue`. Usa `threading.Event` para detener el hilo.
- **Toleras fallos**: descarta los primeros ~5 frames. Declara la cámara desconectada solo tras unos 30 fallos seguidos (~1 s). Después, reconecta con backoff de 0,5 a 5 s.
- Expón un estado `enum`: `CONNECTING`, `LIVE`, `RECONNECTING`, `FAILED`. La UI lo muestra.
- **El propio hilo de captura llama a `cap.release()`** al salir. `stop()` solo activa el `Event` y hace `join`.
- Al arrancar, sondea los índices 0–3 en segundo plano y lista "Cámara N (ancho×alto)". Guarda la última cámara elegida.
- Permisos: en macOS el permiso pertenece a la app anfitriona (VS Code, Terminal o la app empaquetada). Si la cámara no abre, o los frames llegan totalmente negros durante más de 2 s, muestra los pasos: Ajustes del Sistema → Privacidad y seguridad → Cámara, más un botón **Reintentar**. Si empaquetas la app, incluye `NSCameraUsageDescription`.

## Procesamiento para máxima legibilidad (`engine.py`)

Orden del pipeline (validado experimentalmente):
1. Recorte de rostro opcional: YuNet a 320 px, cada 3 frames, con bbox suavizado por EMA, 40 % de margen y relación de aspecto conservada.
2. Gris y reducción a una **resolución de trabajo** de unos 320 px de ancho con `INTER_AREA`.
3. **CLAHE a esa resolución** (clip 2,0, tiles 4×4).
4. Realce de contornos opcional: magnitud de Sobel mezclada con un deslizador de "Bordes".
5. Reducción a la grilla de celdas (`cols × rows`) con `INTER_AREA`. La relación de aspecto de celda se calcula con `QFontMetrics` (alto / avance horizontal), no con 0,5 fijo.
6. **Auto-niveles por percentiles** (2–98) suavizados en el tiempo, en lugar de `NORM_MINMAX` por frame.
7. **EMA temporal** sobre `float32` (α ≈ 0,4, configurable). Opcional: histéresis de medio nivel.
8. Gamma y brillo, después cuantización a N niveles: índices `uint8`.
9. Espejo aplicado a la grilla, que es barato.

Garantiza que con cualquier tamaño (incluidos 0 o 1 filas o columnas) no haya excepción: tamaño mínimo 1×1.

Modos de render (un selector):
- **ASCII**: paleta por defecto de 10–16 caracteres. Las paletas se **ordenan midiendo la tinta real** de cada glifo, renderizándolo con QPainter en la fuente elegida, y se eligen N caracteres equiespaciados en cobertura. Presets: Simple (10), Detallada (16), Extendida (≈70, solo para tamaños grandes).
- **Bloques**: `" ░▒▓█"`. En modo color, usa medio bloque `▀` con color de primer plano y fondo, lo que da 2 píxeles verticales por celda.
- **Braille**: 2×4 puntos por celda (8× más resolución espacial) con dithering ordenado de Bayer. Genera los 256 glifos **de forma procedural** en el atlas, sin depender de la fuente.
- **Bordes**: usa `| / - \` donde el gradiente es fuerte (según la dirección de Sobel), y la paleta normal en el resto.

Color: Mono, Fósforo (verde) o Color real. En Color real, cada glifo se tiñe con el color medio de su celda, con un ligero aumento de saturación.

## Render en la UI (`render.py`, `ui/viewport.py`)

- **Atlas de glifos**: un array `(N, alto_celda, ancho_celda)` `uint8` precalculado. Compón el frame con indexación numpy: `atlas[idx].transpose(0,2,1,3).reshape(H*ch, W*cw)`. No dibujes texto carácter a carácter con QPainter ni uses `QTextEdit`.
- Pasa a `QImage` con stride explícito y conserva una referencia al buffer numpy mientras el `QImage` exista (o usa `.copy()`). Escala para ajustar al visor, centrado y conservando el aspecto.
- Exportar texto: tabla de bytes `np.frombuffer(paleta.encode(), np.uint8)[idx]` más una columna de `\n` y después `.tobytes().decode()`.

## Concurrencia en Qt

- Un `QTimer` de ~33 ms en el hilo de UI toma el último frame. Si `seq` no cambió, no hace nada. Si cambió, procesa y pinta.
- Si el procesamiento medido supera 16 ms, muévelo a un `QThread` worker con señales.
- **Nunca toques widgets desde otro hilo**, y no bloquees el hilo de UI con E/S de cámara.

## Diseño de la interfaz

```
┌──────────────────────────────────────────────┬───────────────────┐
│                                              │ FUENTE            │
│                                              │ Cámara  [Cám 0 ▼] │
│                                              │ Espejo  [✓]       │
│               VISOR ASCII                    │ IMAGEN            │
│     (escalado para ajustarse, centrado)      │ Contraste ───●──  │
│                                              │ Gamma     ──●───  │
│                                              │ Bordes    ●─────  │
│                                              │ ESTILO            │
│                                              │ Modo   [ASCII ▼]  │
│                                              │ Paleta [Simple ▼] │
│                                              │ Color  [Mono ▼]   │
│                                              │ Detalle ──●── 120 │
│                                              │ ENCUADRE          │
│                                              │ Seguir rostro [ ] │
│                                              │ Estabilidad ──●── │
├──────────────────────────────────────────────┴───────────────────┤
│ ● En vivo · Cámara 0 1280×720 · 120×40 celdas · 30 fps · 6 ms    │
└──────────────────────────────────────────────────────────────────┘
```

- Tema oscuro por defecto (fondo cercano a `#0d0f12`, texto `#d8dee9`), con buen contraste. El panel lateral se puede plegar.
- Cada deslizador muestra su valor numérico. Los cambios se aplican en vivo, sin reiniciar la cámara. Botón **Restablecer ajustes**.
- Barra de estado: estado de la cámara, resolución real, tamaño de la grilla, fps y ms de procesamiento.
- Vista dividida Original | ASCII (tecla `O`) para comparar.
- Acciones y atajos: Pausa (`Espacio`), Guardar PNG + TXT (`S`), Copiar ASCII (`Cmd+C`), Pantalla completa (`F`). Cada control tiene tooltip.
- Estados vacíos y de error claros: "Sin permiso de cámara" (con los pasos), "Reconectando…" como overlay y "Modelo de rostro no encontrado" (función desactivada, sin cerrar la app).
- Los ajustes persisten con `QSettings`.

## Reglas de codificación

- Type hints, `@dataclass` para la configuración, funciones pequeñas y puras en `engine`.
- Sin estado global mutable. Constantes con nombre. Comentarios en español.
- `import cv2` normal dentro de `try`, con un mensaje claro si falta (sin `importlib.import_module`).
- `logging` en lugar de `print` (salvo en el renderer de terminal).
- No descargues modelos en silencio: documenta en el README de dónde obtener el `.onnx`.
- No añadas dependencias más allá de PySide6-Essentials, opencv-python-headless, numpy y pytest.

## Pruebas y criterios de aceptación

`tests/test_engine.py` (pytest, frames sintéticos, sin cámara):
- Tamaños (120,40), (80,1), (80,0), (1,1), (400,120): sin excepciones y con la forma esperada.
- Un gradiente horizontal produce índices no decrecientes (mapeo monótono).
- **Parpadeo**: escena estática con ruido σ=3 durante 40 frames: **< 5 %** de celdas cambian por frame en el modo por defecto.
- **Memoria**: 3.000 frames 720p con `tracemalloc`: crecimiento < 1 MB.
- **Rendimiento**: procesamiento más render de un frame 720p a 120 columnas **< 10 ms** en este Mac.
- La paleta medida por cobertura es estrictamente creciente en tinta.

`tests/test_capture.py` (captura falsa, sin cámara real):
- Un fallo transitorio de `read()` **no** detiene el stream.
- 30 o más fallos seguidos pasan a `RECONNECTING` y luego a `LIVE` al recuperarse.
- `stop()` termina en < 1 s, y `release()` se llama desde el hilo de captura.

Prueba manual (describe lo que observaste):
- Arranque, cambio entre las dos cámaras y todos los modos.
- Denegar el permiso muestra el mensaje con los pasos.
- 10 minutos en marcha con memoria estable (`ps -o rss`).
- `python facia.py` sigue funcionando en la terminal.

## Fases y entregables

1. `engine.py` + tests (corrige el parpadeo, la paleta y los tamaños). Reporta el % de parpadeo antes y después.
2. `capture.py` + tests (tolerancia a fallos, reconexión, release en el hilo propio).
3. `terminal.py` y `facia.py` delegando en el paquete.
4. UI PySide6 con visor, panel, estado y atajos.
5. Modos Bloques, Braille, Bordes, Color real y seguimiento de rostro.
6. README breve: instalación con venv, ejecución, permisos de cámara en macOS y origen del modelo YuNet.

Al final, entrega un resumen con lo implementado, los resultados de las pruebas (números reales) y cualquier punto pendiente o riesgo.
