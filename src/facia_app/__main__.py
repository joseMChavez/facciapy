"""Punto de entrada: `facia` o `python -m facia_app`."""

import argparse
import logging
import sys

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="facia", description="Cámara a arte ASCII")
    parser.add_argument("--terminal", action="store_true", help="modo consola (sin interfaz)")
    parser.add_argument("--camera", type=int, default=None, help="índice de cámara")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.terminal:
        from . import terminal
        terminal.run(Settings(camera_index=args.camera or 0))
        return

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        raise SystemExit('Falta PySide6. Instala con: pip install -e ".[ui]"') from None
    from .ui.main_window import MainWindow, load_settings

    app = QApplication(sys.argv)
    settings = load_settings()
    if args.camera is not None:
        settings.camera_index = args.camera
    win = MainWindow(settings)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
