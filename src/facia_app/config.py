"""Configuración de la aplicación."""

from dataclasses import dataclass


MODES = ("ASCII", "Bloques", "Braille", "Bordes")


@dataclass
class Settings:
    camera_index: int = 0
    mirror: bool = True
    gamma: float = 0.8
    smoothing: float = 0.4  # alfa de la EMA temporal (1 = sin suavizado)
    palette: str = " .:-=+*#%@"  # paleta fija del modo consola
    char_aspect: float = 0.5  # ancho/alto de celda (modo consola)
    # Solo interfaz gráfica
    palette_name: str = "Simple"
    theme: str = "Mono"
    detail: int = 120  # columnas de la rejilla
    mode: str = "ASCII"
    edge_sensitivity: int = 50  # 0..100, solo modo Bordes
    split_view: bool = False  # Original | ASCII
