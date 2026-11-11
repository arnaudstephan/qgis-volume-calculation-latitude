# -*- coding: utf-8 -*-
"""French translation: complete, consistent with the code, compiled, and really used by the dialog.

Run with the QGIS Python from the repository root:
    "C:\\Program Files\\QGIS 3.44.0\\bin\\python-qgis.bat" volume_calculation_tool/test/test_translation.py
"""
import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN))
sys.path.insert(0, os.path.join(PLUGIN, "i18n"))

from qgis.PyQt.QtCore import QCoreApplication, QSettings  # noqa: E402
from qgis.core import QgsApplication  # noqa: E402

TMP = tempfile.mkdtemp(prefix="voli18n_")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TMP)
_APP = QgsApplication([], True)
_APP.initQgis()
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TMP)

import i18n_tool  # noqa: E402
from volume_calculation_tool import translation  # noqa: E402
from volume_calculation_tool.volume_engine import BaseLevelOptions, CountOptions  # noqa: E402


class TranslationFilesTest(unittest.TestCase):

    def test_every_text_translated_with_same_placeholders(self):
        contexts = i18n_tool.extract()
        translations = i18n_tool.read_ts(os.path.join(PLUGIN, "i18n", i18n_tool.TS_PATTERN.format("fr")))
        self.assertGreater(sum(len(t) for t in contexts.values()), 100)
        self.assertEqual(i18n_tool.problems(contexts, translations), [])

    def test_compiled_file_is_up_to_date(self):
        shipped = os.path.join(PLUGIN, "i18n", i18n_tool.QM_PATTERN.format("fr"))
        self.assertTrue(os.path.exists(shipped), "run: python i18n/i18n_tool.py build")
        rebuilt = os.path.join(TMP, "fr.qm")
        translations = i18n_tool.read_ts(os.path.join(PLUGIN, "i18n", i18n_tool.TS_PATTERN.format("fr")))
        i18n_tool.write_qm(rebuilt, "fr", translations)
        with open(shipped, "rb") as a, open(rebuilt, "rb") as b:
            self.assertEqual(a.read(), b.read(), "the .qm is older than the .ts: run i18n_tool.py build")


class FrenchInterfaceTest(unittest.TestCase):

    def setUp(self):
        self.translator = translation.install_translator("fr")
        self.assertIsNotNone(self.translator)

    def tearDown(self):
        translation.remove_translator(self.translator)

    def test_texts_of_the_engine(self):
        self.assertEqual(BaseLevelOptions.MANUAL_BASE_LEVEL.short_label, "Altitude imposée")
        self.assertEqual(BaseLevelOptions.APPROXIMATE_VIA_MIN.short_label, "Contour – point le plus bas")
        self.assertEqual(str(CountOptions.COUNT_ONLY_ABOVE), "Au-dessus de la base (stock)")
        self.assertIn("anneau", BaseLevelOptions.SPLINE_RING.explanation)

    def test_dialog_in_french(self):
        from volume_calculation_tool.volume_calculation_tool_dialog import VolumeCalculationToolDialog
        dlg = VolumeCalculationToolDialog()
        try:
            self.assertEqual(dlg.windowTitle(), "Calcul de volumes")
            self.assertEqual(dlg.pushButtonReloadPolygons.text(), "Recharger")
            self.assertEqual(dlg.tablePolygons.horizontalHeaderItem(3).text(), "Méthode de base")
            self.assertEqual(dlg.groupBoxPolygons.title(), "Volumes par polygone")
            self.assertIn("Calculer", dlg.pushButtonStartCalculation.text())
            counting = [dlg.mFieldComboCountingMethod.itemText(i) for i in range(dlg.mFieldComboCountingMethod.count())]
            self.assertIn("Déblai et remblai", counting)
        finally:
            dlg.close()
            dlg.deleteLater()
            QCoreApplication.processEvents()

    def test_english_without_translator(self):
        translation.remove_translator(self.translator)
        try:
            self.assertEqual(BaseLevelOptions.MANUAL_BASE_LEVEL.short_label, "Manual level")
        finally:
            self.translator = translation.install_translator("fr")


if __name__ == "__main__":
    unittest.main(verbosity=2)
