# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 Latitude Drone - GNU GPL v2 or later (see LICENSE).
Translation of the plugin.

The source texts are in English. The translations are Qt files in the i18n folder:
VolumeCalculationTool_<language>.ts (editable with Qt Linguist or by hand) compiled into
VolumeCalculationTool_<language>.qm (i18n/i18n_tool.py build). The language is the one of the
QGIS user interface; without a .qm file for it, the plugin stays in English.

In the code:
    tr("text")        translated text, for texts used at once (messages, widgets, log)
    tr_noop("text")   marks a module constant for translation; it is translated with tr()
                      where it is used (the translator is not installed yet at import time)
The .ui texts are translated by Qt itself (context of the form class).
"""

import os

from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator

CONTEXT = "VolumeCalculationTool"
I18N_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n")
QM_PATTERN = "VolumeCalculationTool_{}.qm"


def tr(text):
    """Translation of an English source text (the text itself without translation)."""
    return QCoreApplication.translate(CONTEXT, text)


def tr_noop(text):
    """Marks `text` for the translation files; translate it with tr() where it is used."""
    return text


def ui_language():
    """Two-letter language of the QGIS user interface ('en' by default)."""
    locale = ""
    try:
        from qgis.core import QgsApplication
        locale = QgsApplication.locale() or ""
    except Exception:
        pass
    if not locale:
        locale = QSettings().value("locale/userLocale", "") or ""
    return (locale or "en")[0:2].lower()


def install_translator(language=None):
    """Install the translator of `language` (default: the QGIS one). Returns it, or None
    when there is no translation for this language (the plugin then stays in English)."""
    path = os.path.join(I18N_DIR, QM_PATTERN.format(language or ui_language()))
    if not os.path.exists(path):
        return None
    translator = QTranslator()
    if not translator.load(path):
        return None
    QCoreApplication.installTranslator(translator)
    return translator


def remove_translator(translator):
    if translator is not None:
        QCoreApplication.removeTranslator(translator)
