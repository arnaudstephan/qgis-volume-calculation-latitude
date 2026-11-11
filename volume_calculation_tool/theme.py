# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 Latitude Drone - GNU GPL v2 or later (see LICENSE).
Look of the dialogs: one accent colour (the blue of the base surface) and a style sheet built
on the palette roles, so the dialogs follow the light and the dark QGIS themes.
"""

from qgis.PyQt.QtGui import QColor, QPalette

ACCENT = "#2168d6"           # blue of the base surface (plugin icon, 3D view)
ACCENT_HOVER = "#2f78e6"
ACCENT_PRESSED = "#1a55b0"
ERROR_COLOR = "#d0312d"

# palette() roles, and the card colour derived from the palette (no fixed background): readable
# with the light and the dark QGIS themes
_STYLE = """
QGroupBox {
    font-weight: 600;
    border: 1px solid palette(midlight);
    border-radius: 6px;
    background: %(card)s;
    margin-top: 1.3em;
    padding: 0px 4px 4px 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 2px;
    padding: 0 2px 2px 2px;
}
QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QComboBox, QGroupBox QAbstractSpinBox,
QGroupBox QLineEdit, QGroupBox QPushButton, QGroupBox QToolButton, QGroupBox QTableView {
    font-weight: normal;
}
QTableView {
    border: 1px solid palette(midlight);
    border-radius: 4px;
    selection-background-color: palette(highlight);
}
QHeaderView::section {
    background: palette(window);
    border: none;
    border-bottom: 1px solid palette(mid);
    padding: 4px 6px;
    font-weight: 600;
}
QPushButton#pushButtonStartCalculation {
    background: %(accent)s;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 7px 18px;
    font-weight: 600;
}
QPushButton#pushButtonStartCalculation:hover { background: %(hover)s; }
QPushButton#pushButtonStartCalculation:pressed { background: %(pressed)s; }
QPushButton#pushButtonStartCalculation:disabled { background: palette(midlight); color: palette(mid); }
QProgressBar {
    border: 1px solid palette(midlight);
    border-radius: 5px;
    background: palette(base);
    text-align: center;
    min-height: 22px;
}
QProgressBar::chunk {
    background: rgba(33, 104, 214, 0.35);
    border-radius: 4px;
}
QTextEdit#logOutput {
    border: 1px solid palette(midlight);
    border-radius: 6px;
    background: transparent;
    padding: 4px;
}
QToolButton#pushButtonPreview3D, QToolButton#toolButtonSetMethod { padding: 3px 8px; }
QToolButton#toolButtonSetMethod { padding-right: 20px; }
QToolButton#toolButtonSetMethod::menu-indicator { subcontrol-origin: padding; subcontrol-position: center right; right: 6px; }
"""


def style_sheet(widget):
    """Style sheet of the main dialog, for the palette of `widget`."""
    return _STYLE % {"accent": ACCENT, "hover": ACCENT_HOVER, "pressed": ACCENT_PRESSED,
                     "card": colors(widget)["card"]}


def mix(first, second, ratio):
    """Colour between `first` (ratio 0) and `second` (ratio 1)."""
    first, second = QColor(first), QColor(second)
    return QColor(int(round(first.red() + (second.red() - first.red()) * ratio)),
                  int(round(first.green() + (second.green() - first.green()) * ratio)),
                  int(round(first.blue() + (second.blue() - first.blue()) * ratio)))


def colors(widget):
    """Colours of the widget's palette, as '#rrggbb' (for rich texts and matplotlib)."""
    palette = widget.palette()
    text = palette.color(QPalette.ColorRole.WindowText)
    window = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    dark = window.lightness() < 128
    # sections: white on a light theme; on a dark theme a little lighter than the window,
    # so that the fields (base colour) stay visible on them
    card = mix(window, text, 0.05) if dark else base
    return {
        "card": card.name(),
        "text": text.name(),
        "muted": mix(text, window, 0.45).name(),
        "window": window.name(),
        "base": base.name(),
        "border": mix(text, card, 0.82).name(),
        "grid": mix(text, card, 0.88).name(),
        "dark": dark,
    }
