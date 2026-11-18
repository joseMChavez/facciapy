import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    """Los tests cierran ventanas, y cada cierre guarda QSettings: nunca en la config real."""
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        yield
        return
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "qs"))
    yield
