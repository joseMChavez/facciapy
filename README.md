# facia-app

Convierte la cámara web en arte ASCII. Modo consola y (en desarrollo) interfaz gráfica con PySide6.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"
```

## Uso

```bash
facia --terminal        # consola
facia                   # interfaz gráfica
facia --camera 1        # elegir cámara
pytest
```

En macOS el permiso de cámara pertenece a la app que lanza Python (Terminal, VS Code…):
Ajustes del Sistema → Privacidad y seguridad → Cámara.

## Arquitectura

```
src/facia_app/
  __main__.py   punto de entrada (CLI)
  config.py     Settings (dataclass)
  capture.py    CameraSource: hilo de captura, último frame, estados
  engine.py     funciones PURAS numpy/cv2: frame -> índices (sin Qt)
  terminal.py   renderer de consola
  glyphs.py     atlas de glifos ordenados por tinta medida (Qt)
  render.py     índices -> imagen/texto
  ui/           ventana, visor y panel de ajustes (PySide6)
tests/          pytest con frames sintéticos, sin cámara
models/         modelos opcionales (YuNet .onnx, no versionado)
docs/PROMPT.md  especificación completa y plan por fases
docs/facia_legacy.py  script original de referencia
```

Hoja de ruta: ver [docs/PROMPT.md](docs/PROMPT.md).

## Modos de render

ASCII · Bloques · Braille (2×4 puntos por celda, 8× más resolución, dithering Bayer) · Bordes (`| / - \\` según el gradiente de Sobel). Combinables con los colores Mono, Fósforo, Ámbar y Color real.

## Imágenes

`Archivo → Abrir imagen…` (Cmd+O), el botón del panel o arrastrar un archivo sobre la ventana. Formatos: PNG, JPG, BMP, WebP, GIF (primer fotograma) y TIFF. Respeta la orientación EXIF; la transparencia se compone sobre negro. «Usar cámara» vuelve a la cámara.

## Exportar

Botón **Exportar imagen…** del panel, tecla `S` o `Archivo → Exportar imagen…`: PNG (por defecto) o JPEG, a escala 1×–4×. A mayor escala los glifos se **re-renderizan** con una fuente más grande (texto nítido, no un reescalado). Opcionalmente guarda también el `.txt`.

**PNG es lo recomendado**: en pruebas pesó entre 10 y 100 veces menos que JPEG (p. ej. 44 KB frente a 1,8 MB a 3480×1960 en Mono/ASCII) y no deja halos en los caracteres. JPEG se ofrece solo por compatibilidad. Límite de 40 MP por exportación.

## Seguimiento de rostro

Opcional (casilla «Seguir rostro»). Usa el detector YuNet de OpenCV; **el modelo no se incluye ni se descarga solo**:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

También vale la variable `FACIA_FACE_MODEL=/ruta/al/modelo.onnx`. Sin el modelo la casilla queda desactivada con un aviso. Detecta a 320 px cada 3 frames, recorta conservando la proporción y mueve el encuadre con suavizado; si pierde el rostro vuelve poco a poco al frame completo.

## Atajos

Espacio pausa · S exportar · Cmd+C copiar ASCII · F pantalla completa · P ocultar panel · O vista dividida Original | ASCII
