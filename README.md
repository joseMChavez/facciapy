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

## Atajos

Espacio pausa · S guardar PNG+TXT · Cmd+C copiar ASCII · F pantalla completa · P ocultar panel
