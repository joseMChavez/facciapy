import importlib
import queue
import shutil
import sys
import threading

try:
    cv2 = importlib.import_module("cv2")
    np = importlib.import_module("numpy")
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is required. Install it with: python -m pip install opencv-python") from exc

# Paleta ordenada de MÁS CLARO a MÁS DENSO (monótona, sin caracteres repetidos).
# Se asume terminal con fondo oscuro: píxel oscuro -> espacio, píxel claro -> carácter denso.
# Si tu terminal tiene fondo claro, pon INVERT = True.
ASCII_CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
INVERT = False

# Tabla de búsqueda: cada valor 0-255 -> índice de carácter (se calcula una sola vez)
_chars = np.array(list(ASCII_CHARS if not INVERT else ASCII_CHARS[::-1]))
LUT = _chars[(np.arange(256) / 255 * (len(_chars) - 1)).astype(int)]

# CLAHE: mejora el contraste local (rasgos de la cara mucho más definidos)
CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
# Corrección gamma (<1 aclara medios tonos, útil porque el ASCII tiende a verse oscuro)
GAMMA_TABLE = (np.linspace(0, 1, 256) ** 0.8 * 255).astype(np.uint8)

CHAR_ASPECT = 0.5  # una celda de terminal es ~2x más alta que ancha


def convert_frame_to_ascii(frame, new_width, max_height):
    h, w = frame.shape[:2]
    new_height = max(1, int(new_width * (h / w) * CHAR_ASPECT))

    # Si no cabe en alto, reducir ancho manteniendo proporción
    if new_height > max_height:
        new_height = max_height
        new_width = max(1, int(new_height / (h / w) / CHAR_ASPECT))

    # INTER_AREA promedia píxeles al reducir: menos ruido y más nitidez
    small = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    gray = CLAHE.apply(gray)               # contraste local
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # usar todo el rango
    gray = cv2.LUT(gray, GAMMA_TABLE)      # gamma

    # Vectorizado con numpy (antes: bucle Python por cada píxel)
    chars = LUT[gray]
    return "\n".join("".join(row) for row in chars)


class CameraStream:
    """Captura de video en un hilo separado (evita lag y bloqueos)."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara (revisa permisos o el índice).")
        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout=1.0):
        # Con timeout para no quedar bloqueado para siempre si la cámara falla
        while True:
            try:
                return self.q.get(timeout=timeout)
            except queue.Empty:
                if self.stopped:
                    return None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


def main():
    try:
        cam = CameraStream(0)
    except Exception as e:
        print(f"Error al iniciar la cámara: {e}")
        return

    out = sys.stdout
    out.write("\033[2J\033[?25l")  # limpiar pantalla y ocultar cursor
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("\nLa cámara dejó de enviar imagen.")
                break

            frame = cv2.flip(frame, 1)  # efecto espejo

            # Se adapta al tamaño actual de la terminal (deja 1 línea libre)
            cols, rows = shutil.get_terminal_size((120, 40))
            art = convert_frame_to_ascii(frame, new_width=cols, max_height=rows - 1)

            # Un solo write + \033[J borra restos si la terminal cambia de tamaño
            out.write("\033[H" + art + "\033[J")
            out.flush()
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\033[?25h\nSaliendo del programa...\n")  # restaurar cursor
        cam.stop()


if __name__ == "__main__":
    main()
