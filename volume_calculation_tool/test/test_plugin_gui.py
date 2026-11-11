# -*- coding: utf-8 -*-
"""End-to-end test of the plugin dialog (offscreen GUI, real QGIS task manager).

Run with the QGIS Python from the repository root:
    "C:\\Program Files\\QGIS 3.44.0\\bin\\python-qgis.bat" volume_calculation_tool/test/test_plugin_gui.py
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from osgeo import gdal, ogr, osr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from qgis.PyQt.QtCore import QCoreApplication, QSettings, Qt  # noqa: E402
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer, QgsVectorLayer  # noqa: E402

TMP = tempfile.mkdtemp(prefix="volgui_")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TMP)
_APP = QgsApplication([], True)
_APP.initQgis()
QCoreApplication.setOrganizationName("VolumeToolTest")
QCoreApplication.setApplicationName("VolumeToolTest")
# QgsApplication re-targets QSettings to its standalone profile: point it back to TMP
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TMP)

# processing providers (PDAL) as in QGIS desktop
_PLUGINS = os.path.join(QgsApplication.prefixPath(), "python", "plugins")
if _PLUGINS not in sys.path:
    sys.path.append(_PLUGINS)
from processing.core.Processing import Processing  # noqa: E402
Processing.initialize()

from test_volume_engine import SIZE, box, square, write_dem  # noqa: E402
from volume_calculation_tool import translation  # noqa: E402
# the assertions are on the English texts, whatever the language of the OS / QGIS
# (the French interface is tested in test_translation.py)
translation.ui_language = lambda: "en"
from volume_calculation_tool.volume_calculation_tool import VolumeCalculationTool  # noqa: E402
from volume_calculation_tool.volume_calculation_tool_dialog import (  # noqa: E402
    COL_ABOVE, COL_BELOW, COL_LEVEL, COL_METHOD, COL_USE, ORIGIN_CHANGED, ORIGIN_STORED)
from volume_calculation_tool.volume_engine import BaseLevelOptions  # noqa: E402


class FakeIface:
    def mainWindow(self):
        return None

    def addToolBarIcon(self, action):
        pass

    def removeToolBarIcon(self, action):
        pass

    def addPluginToRasterMenu(self, menu, action):
        pass

    def removePluginRasterMenu(self, menu, action):
        pass


def make_gpkg(path, polygons, epsg=32632):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    layer = ds.CreateLayer("areas", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
    for name, geom in polygons:
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("name", name)
        feat.SetGeometry(ogr.CreateGeometryFromWkt(geom.asWkt()))
        layer.CreateFeature(feat)
    ds = None


def make_las(path, epsg=32632, vegetation=True):
    """Ground points on a 0.5 m jittered grid: flat 100 m + paraboloid pile (R=10 m, h=5 m)
    centred at local (25, 25), exact volume pi*R^2*h/2 = 785.40 m3. Optional class 5
    'vegetation' points 8 m above ground over the pile, to test the ground filter."""
    import json
    import subprocess
    rng = np.random.default_rng(1)
    gx, gy = np.meshgrid(np.arange(0, 50, 0.5), np.arange(0, 50, 0.5))
    x = (gx + rng.uniform(-0.2, 0.2, gx.shape)).ravel().clip(0, 50)
    y = (gy + rng.uniform(-0.2, 0.2, gy.shape)).ravel().clip(0, 50)
    r2 = (x - 25) ** 2 + (y - 25) ** 2
    z = 100 + np.where(r2 < 100, 5 * (1 - r2 / 100), 0)
    cls = np.full(x.shape, 2)
    if vegetation:
        vx = rng.uniform(18, 32, 300)
        vy = rng.uniform(18, 32, 300)
        x = np.concatenate([x, vx])
        y = np.concatenate([y, vy])
        z = np.concatenate([z, np.full(300, 108.0)])
        cls = np.concatenate([cls, np.full(300, 5)])
    csv = path + ".csv"
    with open(csv, "w") as handle:
        handle.write("X,Y,Z,Classification\n")
        for xi, yi, zi, ci in zip(x + 500000.0, y + 5000000.0, z, cls):
            handle.write("{:.3f},{:.3f},{:.3f},{}\n".format(xi, yi, zi, ci))
    pipeline = [{"type": "readers.text", "filename": csv},
                {"type": "writers.las", "filename": path, "a_srs": "EPSG:{}".format(epsg),
                 "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001,
                 "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}]
    pipe_file = path + ".json"
    with open(pipe_file, "w") as handle:
        json.dump(pipeline, handle)
    subprocess.run([shutil.which("pdal"), "pipeline", pipe_file], check=True)


class PluginGuiTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().clear()
        shutil.rmtree(TMP, ignore_errors=True)

    def setUp(self):
        QgsProject.instance().clear()
        # every test starts from the default plugin preferences
        from volume_calculation_tool.volume_calculation_tool_dialog import SETTINGS_PREFIX
        QSettings().remove(SETTINGS_PREFIX.rstrip("/"))
        ground = np.full((SIZE, SIZE), 100.0)
        z = box(ground, 5, 5, 10, 5.0)     # pile A: 500 m3 above 100 m
        z = box(z, 30, 30, 10, 3.0)        # pile B: +3 m
        stamp = "{}_{}".format(self._testMethodName[-12:], int(time.time() * 1000))
        self.dem_path = os.path.join(TMP, "dem.tif") if "ndom" in self._testMethodName \
            else os.path.join(TMP, "dem_{}.tif".format(stamp))
        write_dem(self.dem_path, z)
        gpkg = os.path.join(TMP, "areas_{}.gpkg".format(stamp))
        make_gpkg(gpkg, [("pile A", square(2, 2, 16)), ("pile B", square(27, 27, 16))])
        self.dem = QgsRasterLayer(self.dem_path, "dem")
        self.poly = QgsVectorLayer(gpkg + "|layername=areas", "areas", "ogr")
        self.assertTrue(self.dem.isValid() and self.poly.isValid())
        self.poly.setDisplayExpression('"name"')
        QgsProject.instance().addMapLayers([self.dem, self.poly])

        self.plugin = VolumeCalculationTool(FakeIface())
        self.plugin.initGui()
        self.messages = []

    def tearDown(self):
        self.plugin.unload()

    def open_dialog(self):
        self.plugin.run()
        dlg = self.plugin.dlg
        dlg.popWarningBox = lambda msg: self.messages.append(("warning", msg))
        dlg.popFatalErrorBox = lambda msg: self.messages.append(("error", msg))
        dlg.askOkCancel = lambda title, text: True
        dlg.mFieldComboHeightLayer.setLayer(self.dem)
        dlg.mFieldComboPolygon.setLayer(self.poly)
        return dlg

    def run_and_wait(self, timeout=60):
        self.plugin.workflow()
        self.assertIsNotNone(self.plugin.task, self.messages)
        start = time.time()
        while self.plugin.task is not None:
            QCoreApplication.processEvents()
            time.sleep(0.01)
            self.assertLess(time.time() - start, timeout, "task did not finish")

    def set_row(self, dlg, row, method, level=None):
        combo = dlg.tablePolygons.cellWidget(row, COL_METHOD)
        combo.setCurrentIndex(combo.findData(method.code))
        if level is not None:
            dlg.tablePolygons.cellWidget(row, COL_LEVEL).setValue(level)

    def values(self, field):
        layer = QgsVectorLayer(self.poly.source(), "check", "ogr")
        return {f["name"]: f[field] for f in layer.getFeatures()}

    def test_per_polygon_methods_fields_labels_ndom(self):
        dlg = self.open_dialog()
        self.assertEqual(dlg.tablePolygons.rowCount(), 2)
        self.assertEqual(dlg.doubleSpinBoxSampleStepX.value(), 1.0)  # default = DEM pixel
        self.set_row(dlg, 0, BaseLevelOptions.APPROXIMATE_VIA_MIN)
        self.set_row(dlg, 1, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.5)
        dlg.checkBox_add_field.setChecked(True)
        dlg.checkBox_labels.setChecked(True)
        dlg.checkBox_create_ndom.setChecked(True)
        dlg.outputAccuracy.setValue(2)
        dlg.comboOutputUnit.setCurrentIndex(dlg.comboOutputUnit.findData("m3"))

        self.run_and_wait()
        self.assertEqual([m for m in self.messages if m[0] == "error"], [])

        above = self.values("V_above")
        below = self.values("V_below")
        self.assertAlmostEqual(above["pile A"], 500.0, places=2)
        self.assertAlmostEqual(below["pile A"], 0.0, places=2)
        # pile B, base 100.5: box 100 cells * 2.5 m above, 156 cells * 0.5 m below
        self.assertAlmostEqual(above["pile B"], 250.0, places=2)
        self.assertAlmostEqual(below["pile B"], 78.0, places=2)
        self.assertEqual(self.values("Base_meth"), {"pile A": "MIN", "pile B": "MANUAL"})
        self.assertAlmostEqual(self.values("Base_z")["pile B"], 100.5)

        self.assertTrue(self.poly.labelsEnabled())
        expression = self.poly.labeling().settings().fieldName
        self.assertIn('"V_above"', expression)
        self.assertIn('m³', expression)

        ndom_layers = QgsProject.instance().mapLayersByName("dem_nDOM")
        self.assertEqual(len(ndom_layers), 1)
        self.assertIn("Total", dlg.logOutput.toPlainText())

        # choices persist in the layer: a new dialog reads them back
        self.plugin.dlg.close()
        self.plugin.dlg = None
        dlg = self.open_dialog()
        choices = {c["name"]: c for c in dlg.polygonChoices()}
        self.assertEqual(choices["pile B"]["method"], BaseLevelOptions.MANUAL_BASE_LEVEL)
        self.assertAlmostEqual(choices["pile B"]["level"], 100.5)
        self.assertEqual(choices["pile A"]["method"], BaseLevelOptions.APPROXIMATE_VIA_MIN)

        # second run overwrites the nDOM that is loaded in the project
        self.run_and_wait()
        self.assertEqual(len(QgsProject.instance().mapLayersByName("dem_nDOM")), 1)

    def test_written_values_and_label_text(self):
        """What the client sees: field values (unit + rounding) and the exact label text."""
        from qgis.core import QgsExpression, QgsExpressionContext, QgsExpressionContextUtils
        from volume_calculation_tool.volume_engine import CountOptions
        dlg = self.open_dialog()
        self.set_row(dlg, 0, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)
        self.set_row(dlg, 1, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.5)
        dlg.checkBox_create_ndom.setChecked(False)
        dlg.checkBox_labels.setChecked(True)
        dlg.outputAccuracy.setValue(3)
        dlg.comboOutputUnit.setCurrentIndex(dlg.comboOutputUnit.findData("yd3"))

        def label_texts():
            layer = QgsVectorLayer(self.poly.source(), "check", "ogr")
            expression = QgsExpression(self.poly.labeling().settings().fieldName)
            context = QgsExpressionContext()
            context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
            out = {}
            for feature in layer.getFeatures():
                context.setFeature(feature)
                out[feature["name"]] = str(expression.evaluate(context)).replace(",", ".")
            return out

        # cut / fill: both values
        dlg.mFieldComboCountingMethod.setCurrentIndex(
            dlg.mFieldComboCountingMethod.findData(CountOptions.COUNT_ABOVE_AND_BELOW.value))
        self.run_and_wait()
        yd3 = 0.764554857984
        above, below = self.values("V_above"), self.values("V_below")
        self.assertEqual(above["pile A"], round(500.0 / yd3, 3))      # 653.975
        self.assertEqual(above["pile B"], round(250.0 / yd3, 3))
        self.assertEqual(below["pile B"], round(78.0 / yd3, 3))
        labels = label_texts()
        self.assertEqual(labels["pile A"], "▲ {:.3f} yd³\n▼ {:.3f} yd³".format(round(500 / yd3, 3), 0.0))
        self.assertEqual(labels["pile B"], "▲ {:.3f} yd³\n▼ {:.3f} yd³".format(round(250 / yd3, 3), round(78 / yd3, 3)))

        # stock only: the label must not mention anything below
        dlg.mFieldComboCountingMethod.setCurrentIndex(
            dlg.mFieldComboCountingMethod.findData(CountOptions.COUNT_ONLY_ABOVE.value))
        dlg.comboOutputUnit.setCurrentIndex(dlg.comboOutputUnit.findData("m3"))
        dlg.outputAccuracy.setValue(1)
        self.run_and_wait()
        labels = label_texts()
        self.assertEqual(labels["pile A"], "500.0 m³")
        self.assertEqual(labels["pile B"], "250.0 m³")
        self.assertNotIn("▼", "".join(labels.values()))

    def test_tin_and_spline_rows(self):
        dlg = self.open_dialog()
        self.set_row(dlg, 0, BaseLevelOptions.TRIANGULATED_OUTLINE)
        self.set_row(dlg, 1, BaseLevelOptions.SPLINE_RING)
        self.assertEqual(dlg.splineParams(dlg.fidOfRow(1)), (3.0, 4.0))   # defaults before any choice
        dlg.setSplineParams(dlg.fidOfRow(1), 4.0, 4.0)                    # per polygon (3D view)
        dlg.checkBox_create_ndom.setChecked(False)
        self.run_and_wait()
        self.assertEqual([m for m in self.messages if m[0] == "error"], [])
        self.assertEqual(self.values("Base_ring"), {"pile A": None, "pile B": 4.0})
        self.assertEqual(self.values("Base_stiff"), {"pile A": None, "pile B": 4.0})
        above = self.values("V_above")
        self.assertAlmostEqual(above["pile A"], 500.0, places=1)
        self.assertAlmostEqual(above["pile B"], 300.0, places=0)
        self.assertEqual(self.values("Base_meth"), {"pile A": "TIN", "pile B": "SPLINE"})
        # the base of each polygon is in the tooltip of its volume (no longer repeated in the log)
        tip = dlg.tablePolygons.item(dlg.rowOfFid(dlg.fidOfRow(1)), COL_ABOVE).toolTip()
        self.assertIn("ring 4 m, stiffness 4", tip)
        self.assertIn("spline on", tip)
        self.assertIn("Total", dlg.logOutput.toPlainText())
        # read back by a new dialog
        self.plugin.dlg.close()
        self.plugin.dlg = None
        dlg = self.open_dialog()
        self.assertEqual(dlg.splineParams(dlg.fidOfRow(1)), (4.0, 4.0))

    def test_excluded_rows_and_dem_method_without_layer(self):
        dlg = self.open_dialog()
        dlg.tablePolygons.item(0, COL_USE).setCheckState(Qt.CheckState.Unchecked)
        self.set_row(dlg, 1, BaseLevelOptions.USE_DEM_LAYER)
        dlg.mFieldComboHeightLayerBase.setLayer(None)
        dlg.checkBox_create_ndom.setChecked(False)
        self.plugin.workflow()
        self.assertIsNone(self.plugin.task)
        self.assertTrue(any("Base DEM" in m[1] for m in self.messages), self.messages)

    def test_table_is_the_only_method_control(self):
        """Bulk edit, selection propagation, method settings feedback, results in the table."""
        from qgis.PyQt.QtCore import QItemSelectionModel
        from volume_calculation_tool.volume_engine import CountOptions
        dlg = self.open_dialog()
        table = dlg.tablePolygons
        dlg.checkBox_create_ndom.setChecked(False)
        method = lambda row: BaseLevelOptions.from_code(table.cellWidget(row, COL_METHOD).currentData())

        # bulk set with a manual level: every row, the level cell becomes editable
        dlg.setRowsMethod([0, 1], BaseLevelOptions.MANUAL_BASE_LEVEL, level=100.5)
        self.assertEqual([method(0), method(1)], [BaseLevelOptions.MANUAL_BASE_LEVEL] * 2)
        self.assertEqual(table.cellWidget(1, COL_LEVEL).value(), 100.5)
        self.assertEqual(table.cellWidget(0, COL_LEVEL).currentIndex(), 1)   # spin box shown
        self.assertEqual(table.cellWidget(0, COL_METHOD).property("origin"), ORIGIN_CHANGED)

        def choose(row, option):
            """Like a user choosing `option` in the combo of `row` (activated = user choice)."""
            combo = table.cellWidget(row, COL_METHOD)
            index = combo.findData(option.code)
            combo.setCurrentIndex(index)
            combo.activated.emit(index)

        # a method chosen in one row of a multi-row selection goes to the whole selection,
        # and the info line says so
        table.selectAll()
        choose(0, BaseLevelOptions.USE_DEM_LAYER)
        self.assertEqual(method(1), BaseLevelOptions.USE_DEM_LAYER)
        self.assertEqual(table.cellWidget(1, COL_LEVEL).currentIndex(), 0)   # dash, not "0.000"
        self.assertIn("Also applied to ID", dlg.labelMethodInfo.text())
        # changed by code (not a user choice): no propagation
        self.set_row(dlg, 0, BaseLevelOptions.APPROXIMATE_VIA_AVG)
        self.assertEqual(method(1), BaseLevelOptions.USE_DEM_LAYER)

        # 'Manual level' chosen for a selection: the level is asked once, given to every row,
        # and a level typed later in one selected row follows to the others
        dlg._askLevel = lambda rows: 99.0
        choose(0, BaseLevelOptions.MANUAL_BASE_LEVEL)
        self.assertEqual([table.cellWidget(r, COL_LEVEL).value() for r in (0, 1)], [99.0, 99.0])
        table.cellWidget(0, COL_LEVEL).setValue(98.5)
        self.assertEqual(table.cellWidget(1, COL_LEVEL).value(), 98.5)
        self.assertIn("Level also applied", dlg.labelMethodInfo.text())
        # cancelled prompt: no level, shown in red, and the calculation refuses it
        dlg._askLevel = lambda rows: None
        table.clearSelection()
        table.cellWidget(1, COL_LEVEL).setValue(None)
        self.assertIsNone(table.cellWidget(1, COL_LEVEL).value())
        self.plugin.workflow()
        self.assertIsNone(self.plugin.task)
        self.assertIn("Enter the manual level", self.messages[-1][1])
        choose(0, BaseLevelOptions.USE_DEM_LAYER)
        self.set_row(dlg, 1, BaseLevelOptions.USE_DEM_LAYER)

        # base DEM needed but missing: flagged before running, with the number of rows
        dlg.mFieldComboHeightLayerBase.setLayer(None)
        self.assertTrue(dlg.labelBaseDemRequired.isVisibleTo(dlg))
        self.assertTrue(dlg.groupBoxMethodSettings.isVisibleTo(dlg))
        self.assertIn("2 checked rows", dlg.label_11.text())
        table.item(1, COL_USE).setCheckState(Qt.CheckState.Unchecked)
        self.assertIn("1 checked row", dlg.label_11.text())

        # nothing checked: the run button is disabled and says so
        dlg.setIncludedFids(set())
        self.assertFalse(dlg.pushButtonStartCalculation.isEnabled())
        self.assertIn("0 volumes", dlg.pushButtonStartCalculation.text())
        self.assertFalse(dlg.labelBaseDemRequired.isVisibleTo(dlg))
        self.assertFalse(dlg.groupBoxMethodSettings.isVisibleTo(dlg))   # shown only when used
        dlg.setIncludedFids(None)
        self.assertIn("2 volumes", dlg.pushButtonStartCalculation.text())

        # counting "above only": the below field name is not used
        dlg.mFieldComboCountingMethod.setCurrentIndex(
            dlg.mFieldComboCountingMethod.findData(CountOptions.COUNT_ONLY_ABOVE.value))
        self.assertFalse(dlg.fieldName_2.isEnabled())
        dlg.mFieldComboCountingMethod.setCurrentIndex(
            dlg.mFieldComboCountingMethod.findData(CountOptions.COUNT_ABOVE_AND_BELOW.value))
        self.assertTrue(dlg.fieldName_2.isEnabled())

        # results appear in the table and are cleared when the row changes
        dlg.setRowsMethod([0, 1], BaseLevelOptions.MANUAL_BASE_LEVEL, level=100.5)
        dlg.outputAccuracy.setValue(1)
        self.run_and_wait()
        self.assertEqual([m for m in self.messages if m[0] == "error"], [])
        self.assertFalse(table.isColumnHidden(COL_ABOVE))
        name_row = {table.item(r, 2).text(): r for r in range(table.rowCount())}
        above_b = table.item(name_row["pile B"], COL_ABOVE).text().replace(",", ".")
        below_b = table.item(name_row["pile B"], COL_BELOW).text().replace(",", ".")
        self.assertEqual((above_b, below_b), ("250.0", "78.0"))
        self.assertIn("Base: Manual level", table.item(name_row["pile B"], COL_ABOVE).toolTip())
        self.assertEqual(table.cellWidget(0, COL_METHOD).property("origin"), ORIGIN_STORED)
        self.assertEqual(dlg.progressBar.format(), "2/2 computed")
        # total of the table under it
        self.assertTrue(dlg.labelTotal.isVisibleTo(dlg))
        for column in (COL_ABOVE, COL_BELOW):
            total = sum(float(table.item(r, column).text().replace(",", ".")) for r in range(table.rowCount()))
            self.assertIn("{:.1f}".format(total), dlg.labelTotal.text().replace(",", "."))
        # unit / decimals: re-formatted, not cleared
        dlg.outputAccuracy.setValue(2)
        self.assertEqual(table.item(name_row["pile B"], COL_ABOVE).text().replace(",", "."), "250.00")
        dlg.comboOutputUnit.setCurrentIndex(dlg.comboOutputUnit.findData("yd3"))
        self.assertIn("yd³", table.horizontalHeaderItem(COL_ABOVE).text())
        self.assertEqual(table.item(name_row["pile B"], COL_ABOVE).text().replace(",", "."),
                         "{:.2f}".format(250.0 / 0.764554857984))
        # a changed row loses its volumes (and the progress text), the others keep them
        table.cellWidget(name_row["pile B"], COL_LEVEL).setValue(101.0)
        self.assertEqual(table.item(name_row["pile B"], COL_ABOVE).text(), "")
        self.assertNotEqual(table.item(name_row["pile A"], COL_ABOVE).text(), "")
        self.assertEqual(dlg.progressBar.format(), "")

    def test_check_change_clears_spline_volumes_and_errors_shown_once(self):
        from volume_calculation_tool.volume_engine import CountOptions
        dlg = self.open_dialog()
        table = dlg.tablePolygons
        self.set_row(dlg, 0, BaseLevelOptions.SPLINE_RING)
        self.set_row(dlg, 1, BaseLevelOptions.APPROXIMATE_VIA_MIN)
        fids = [table.item(r, 1).data(Qt.ItemDataRole.UserRole) for r in (0, 1)]
        dlg.setResults({fids[0]: {"above_m3": 10.0, "below_m3": 0.0, "tooltip": "", "error": None},
                        fids[1]: {"error": "outline outside the DEM"}},
                       CountOptions.COUNT_ABOVE_AND_BELOW)
        self.assertEqual((table.item(1, COL_ABOVE).text(), table.item(1, COL_BELOW).text()), ("Error", "—"))
        table.setCurrentCell(1, 2)
        self.assertIn("not computed", dlg.labelMethodInfo.text())
        self.assertIn("outline outside the DEM", dlg.labelMethodInfo.text())
        # unchecking another polygon changes the ground ring of the spline row: its volume goes
        table.item(1, COL_USE).setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(table.item(0, COL_ABOVE).text(), "")
        self.assertEqual(table.item(1, COL_ABOVE).text(), "Error")

    def test_step_rounded_by_spin_box_keeps_exact_dem_grid(self):
        """A pixel size the spin box cannot show exactly is still sampled on the DEM grid."""
        path = os.path.join(TMP, "odd_pixel.tif")
        write_dem(path, np.full((SIZE, SIZE), 100.0), pixel=0.04987)
        odd = QgsRasterLayer(path, "odd")
        QgsProject.instance().addMapLayer(odd)
        dlg = self.open_dialog()
        dlg.mFieldComboHeightLayer.setLayer(odd)
        self.assertEqual(dlg.doubleSpinBoxSampleStepX.value(), 0.0499)   # rounded by the spin box
        built = self.plugin.buildCalculation()
        self.assertIsNotNone(built)
        _jobs, settings, ctx = built
        self.assertEqual(settings.step_x, settings.pixel_x)
        self.assertEqual(ctx["step_x"], odd.rasterUnitsPerPixelX())

    def test_reprojected_polygons_and_base_dem(self):
        # polygons in Web Mercator, base DEM (flat 100 m) warped to Web Mercator, DEM in UTM 32N
        flat = os.path.join(TMP, "flat_utm.tif")
        write_dem(flat, np.full((SIZE, SIZE), 100.0))
        flat_3857 = os.path.join(TMP, "flat_3857.tif")
        gdal.Warp(flat_3857, flat, dstSRS="EPSG:3857", resampleAlg="near")
        base = QgsRasterLayer(flat_3857, "base3857")
        self.assertTrue(base.isValid())

        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
        to_3857 = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:32632"),
                                         QgsCoordinateReferenceSystem("EPSG:3857"), QgsProject.instance())
        geom = square(2, 2, 16)
        geom.transform(to_3857)
        gpkg = os.path.join(TMP, "areas_3857.gpkg")
        make_gpkg(gpkg, [("pile A", geom)], epsg=3857)
        poly = QgsVectorLayer(gpkg + "|layername=areas", "areas3857", "ogr")
        poly.setDisplayExpression('"name"')
        QgsProject.instance().addMapLayers([base, poly])
        self.poly = poly

        dlg = self.open_dialog()
        dlg.mFieldComboPolygon.setLayer(poly)
        dlg.mFieldComboHeightLayerBase.setLayer(base)
        dlg.checkBox_create_ndom.setChecked(False)
        for method in (BaseLevelOptions.APPROXIMATE_VIA_MIN, BaseLevelOptions.USE_DEM_LAYER):
            self.set_row(dlg, 0, method)
            self.run_and_wait()
            self.assertEqual([m for m in self.messages if m[0] == "error"], [])
            self.assertAlmostEqual(self.values("V_above")["pile A"], 500.0, places=1, msg=str(method))
        self.assertIn("reprojected", dlg.logOutput.toPlainText())

    def test_las_point_cloud_input(self):
        from qgis.core import QgsPointCloudLayer
        las = os.path.join(TMP, "ground.las")
        make_las(las)
        cloud = QgsPointCloudLayer(las, "cloud", "pdal")
        self.assertTrue(cloud.isValid())
        gpkg = os.path.join(TMP, "pile_las.gpkg")
        make_gpkg(gpkg, [("pile", square(10, 10, 30))])
        # square() works in DEM-local coordinates (top at Y0); the LAS is placed at y 5000000..5000050
        poly = QgsVectorLayer(gpkg + "|layername=areas", "pile_las", "ogr")
        poly.setDisplayExpression('"name"')
        QgsProject.instance().addMapLayers([cloud, poly])
        self.poly = poly

        dlg = self.open_dialog()
        dlg.mFieldComboHeightLayer.setLayer(cloud)
        dlg.mFieldComboPolygon.setLayer(poly)
        self.assertTrue(dlg.widgetPointCloud.isVisibleTo(dlg))
        self.assertAlmostEqual(dlg.doubleSpinBoxCloudResolution.value(), 0.5)  # ~ point spacing
        dlg.doubleSpinBoxCloudResolution.setValue(0.25)
        self.assertEqual(dlg.doubleSpinBoxSampleStepX.value(), 0.25)
        self.set_row(dlg, 0, BaseLevelOptions.APPROXIMATE_VIA_MIN)
        dlg.checkBox_create_ndom.setChecked(False)

        from qgis.core import QgsHillshadeRenderer
        kept_path = os.path.join(TMP, "ground_DEM_TIN_0p25_class2.tif")

        def kept_layers():
            return [l for l in QgsProject.instance().mapLayers().values()
                    if l.source().split("|")[0].replace("\\", "/").lower() == kept_path.replace("\\", "/").lower()]

        results = []
        # 1) ground only, kept DEM  2) same again: reused, layer untouched  3) all points, temporary DEM
        for ground_only, keep in ((True, True), (True, True), (False, False)):
            dlg.checkBoxGroundOnly.setChecked(ground_only)
            dlg.checkBoxLoadCloudDem.setChecked(keep)
            dlg.logOutput.clear()
            self.run_and_wait()               # step 1 (triangulation) then step 2 (volume)
            self.assertEqual([m for m in self.messages if m[0] == "error"], [])
            results.append((self.values("V_above")["pile"], dlg.logOutput.toPlainText()))
            if len(results) == 1:
                layers = kept_layers()
                self.assertEqual(len(layers), 1)
                first_layer_id = layers[0].id()
                self.assertIsInstance(layers[0].renderer(), QgsHillshadeRenderer)
                self.assertTrue(os.path.exists(kept_path[:-4] + ".qml"))
                # the kept DEM covers the whole cloud (0..50 m), not only the polygon (10..40 m)
                self.assertGreater(layers[0].extent().width(), 45)
        self.assertAlmostEqual(results[0][0], 785.40, delta=785.40 * 0.01)
        self.assertAlmostEqual(results[1][0], results[0][0], places=6)
        self.assertIn("reused", results[1][1])
        self.assertEqual([l.id() for l in kept_layers()], [first_layer_id])   # not removed / re-added
        self.assertGreater(results[2][0], results[0][0] + 50)  # vegetation counted without the class filter
        self.assertIn("polygons area only", results[2][1])
        self.assertIn("point cloud 'cloud'", results[0][1])

    def test_label_shows_material_then_volume(self):
        """A field that looks like a material is chosen automatically and written above the volume."""
        from qgis.core import QgsExpression, QgsExpressionContext, QgsExpressionContextUtils, QgsField
        from qgis.PyQt.QtCore import QVariant
        from volume_calculation_tool.volume_engine import CountOptions
        provider = self.poly.dataProvider()
        provider.addAttributes([QgsField("Matériau", QVariant.String)])
        self.poly.updateFields()
        index = self.poly.fields().lookupField("Matériau")
        first = next(f.id() for f in self.poly.getFeatures() if f["name"] == "pile A")
        provider.changeAttributeValues({first: {index: "Gravier 0/31,5"}})   # pile B: no material

        dlg = self.open_dialog()
        self.assertEqual(dlg.fieldComboLabelName.currentField(), "Matériau")
        dlg.checkBox_labels.setChecked(True)
        dlg.checkBox_create_ndom.setChecked(False)
        dlg.outputAccuracy.setValue(1)
        dlg.mFieldComboCountingMethod.setCurrentIndex(
            dlg.mFieldComboCountingMethod.findData(CountOptions.COUNT_ONLY_ABOVE.value))
        self.set_row(dlg, 0, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)
        self.set_row(dlg, 1, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.5)
        self.run_and_wait()

        layer = QgsVectorLayer(self.poly.source(), "check", "ogr")
        expression = QgsExpression(self.poly.labeling().settings().fieldName)
        self.assertFalse(expression.hasParserError(), expression.parserErrorString())
        context = QgsExpressionContext()
        context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
        labels = {}
        for feature in layer.getFeatures():
            context.setFeature(feature)
            labels[feature["name"]] = str(expression.evaluate(context))
        # the volume may use the locale decimal comma; the material is written as typed
        self.assertEqual(labels["pile A"].split("\n")[0], "Gravier 0/31,5")
        self.assertEqual(labels["pile A"].split("\n")[1].replace(",", "."), "500.0 m³")
        self.assertEqual(labels["pile B"].replace(",", "."), "250.0 m³")   # no material: volume only

        # no name field chosen: volume only, as before
        dlg.fieldComboLabelName.setField("")
        self.run_and_wait()
        expression = QgsExpression(self.poly.labeling().settings().fieldName)
        self.assertNotIn("Matériau", expression.expression())

    def test_base_preview_3d(self):
        from volume_calculation_tool.volume_calculation_tool_dialog import COL_ID
        dlg = self.open_dialog()
        self.set_row(dlg, 0, BaseLevelOptions.APPROXIMATE_VIA_MIN)
        dlg.mFieldComboHeightLayerBase.setLayer(None)
        fid = dlg.tablePolygons.item(0, COL_ID).data(Qt.ItemDataRole.UserRole)
        self.plugin.showBasePreview(fid)
        self.assertEqual(self.messages, [])
        self.assertEqual(len(self.plugin.previews), 1)
        preview = self.plugin.previews[0]
        preview.canvas.draw()
        info = lambda: preview.labelInfo.text().replace(",", ".")   # decimal separator of the locale
        self.assertIn("500.0 m³", info())
        self.assertEqual(preview.computer.inside.sum() > 0, True)

        # opening the view changes nothing (the method field does not even exist yet)
        self.assertEqual(QgsVectorLayer(self.poly.source(), "check", "ogr").fields().lookupField("Base_meth"), -1)

        # every method: drawn and kept for the polygon (table + layer at once), or a clear message
        # when it cannot be built (no base DEM) and then the row keeps its previous method
        self.assertFalse(preview.spinLevel.isVisibleTo(preview))   # level field only for 'Manual level'
        for option in (BaseLevelOptions.APPROXIMATE_VIA_AVG, BaseLevelOptions.BEST_FIT_PLANE,
                       BaseLevelOptions.TRIANGULATED_OUTLINE, BaseLevelOptions.SPLINE_RING,
                       BaseLevelOptions.USE_DEM_LAYER, BaseLevelOptions.APPROXIMATE_VIA_MIN):
            preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(option.code))
            preview.canvas.draw()
            text = preview.labelInfo.text()
            if option == BaseLevelOptions.USE_DEM_LAYER:
                self.assertIn("Not possible for this polygon", text, option)
                self.assertEqual(dlg.tablePolygons.cellWidget(0, COL_METHOD).currentData(), "SPLINE")
                self.assertEqual(self.values("Base_meth")["pile A"], "SPLINE")
            else:
                self.assertIsNotNone(preview._data, (option, text))
                self.assertIn("above the base", text)
                self.assertIn("Saved in the layer.", preview.labelNote.text())
                self.assertEqual(dlg.tablePolygons.cellWidget(0, COL_METHOD).currentData(), option.code)
                self.assertEqual(self.values("Base_meth")["pile A"], option.code)
        self.assertEqual(dlg.tablePolygons.cellWidget(0, COL_METHOD).property("origin"), ORIGIN_STORED)

        # spline: ring width and stiffness of this polygon, set here, kept in the layer
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.SPLINE_RING.code))
        self.assertTrue(preview.spinRing.isVisibleTo(preview) and preview.spinStiffness.isVisibleTo(preview))
        self.assertFalse(preview.spinLevel.isVisibleTo(preview))
        self.assertEqual(preview.splineParams(), (3.0, 4.0))
        self.assertEqual(self.values("Base_ring")["pile A"], 3.0)
        preview.spinRing.setValue(12.0)       # wider than the view margin: the view grows
        preview.spinStiffness.setValue(8.0)
        preview._levelTimer.stop()
        preview._levelTimer.timeout.emit()
        preview.canvas.draw()
        self.assertIn("ring 12 m, stiffness 8", preview.labelInfo.text())
        self.assertGreaterEqual(preview.computer.margin, 15.0 - 1e-6)
        self.assertEqual((self.values("Base_ring")["pile A"], self.values("Base_stiff")["pile A"]), (12.0, 8.0))
        self.assertEqual(dlg.splineParams(fid), (12.0, 8.0))
        # sliders follow the fields and drive them (half steps); releasing saves at once
        self.assertEqual((preview.sliderRing.value(), preview.sliderStiffness.value()), (24, 16))
        preview.sliderRing.setValue(13)
        preview.sliderStiffness.setValue(5)
        self.assertEqual(preview.splineParams(), (6.5, 2.5))
        preview.sliderRing.sliderReleased.emit()
        self.assertEqual((self.values("Base_ring")["pile A"], self.values("Base_stiff")["pile A"]), (6.5, 2.5))
        preview.spinRing.setValue(80.0)                   # beyond the slider: field keeps it, slider at max
        self.assertEqual(preview.sliderRing.value(), preview.sliderRing.maximum())
        preview._flushLevel()
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.APPROXIMATE_VIA_MIN.code))
        self.assertFalse(preview.spinRing.isVisibleTo(preview))
        self.assertEqual(self.values("Base_ring")["pile A"], 80.0)     # kept for a later return to the spline
        for mode in range(preview.comboStock.count()):
            preview.comboStock.setCurrentIndex(mode)
            preview.canvas.draw()

        # lighter mesh while the view is dragged, fine mesh again on release
        # (small test DEM: the display grid is already light, no second mesh is built)
        from volume_calculation_tool.base_preview import ROTATION_DISPLAY_NODES

        class _Event:
            inaxes = preview.ax
        if preview.computer.xs.size <= ROTATION_DISPLAY_NODES:
            self.assertIsNone(preview._coarseMesh)
            preview._onMousePress(_Event())
            preview._onMouseRelease(_Event())
            self.assertTrue(preview._fineMesh.get_visible())
        else:
            preview._onMousePress(_Event())
            self.assertTrue(preview._coarseMesh.get_visible() and not preview._fineMesh.get_visible())
            preview._onMouseRelease(_Event())
            self.assertTrue(preview._fineMesh.get_visible() and not preview._coarseMesh.get_visible())

        # manual level typed in the 3D view: no level in the row -> starts at the lowest outline point
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.MANUAL_BASE_LEVEL.code))
        self.assertTrue(preview.spinLevel.isVisibleTo(preview))
        self.assertAlmostEqual(preview.spinLevel.value(), 100.0)
        self.assertIn("500.0 m³", info())
        self.assertAlmostEqual(dlg.tablePolygons.cellWidget(0, COL_LEVEL).value(), 100.0)   # kept at once
        preview.spinLevel.setValue(99.0)      # pile A is 16 x 16 m: +256 m3 one metre lower
        self.assertTrue(preview._levelTimer.isActive())          # taken after a short pause...
        preview._levelTimer.stop()
        preview._levelTimer.timeout.emit()
        self.assertIn("756.0 m³", info())
        self.assertAlmostEqual(self.values("Base_z")["pile A"], 99.0)
        preview.spinLevel.setValue(99.5)      # ... or when leaving the polygon right after typing

        # arrows: next / previous polygon of the table, in the same window, row selected in the table
        fid_b = dlg.fidOfRow(1)
        self.set_row(dlg, 1, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.5)
        preview.buttonNext.click()
        self.assertAlmostEqual(self.values("Base_z")["pile A"], 99.5)
        self.assertAlmostEqual(dlg.tablePolygons.cellWidget(0, COL_LEVEL).value(), 99.5)
        self.assertEqual(preview.computer.fid, fid_b)
        self.assertEqual(dlg.tablePolygons.currentRow(), 1)
        self.assertEqual(preview.currentMethod(), BaseLevelOptions.MANUAL_BASE_LEVEL)   # method of that row
        self.assertAlmostEqual(preview.spinLevel.value(), 100.5)
        self.assertIn("250.0 m³", info())
        self.assertIn(str(fid_b), preview.labelPolygon.text())
        self.assertEqual(self.values("Base_meth")["pile B"], None)   # browsing does not save anything
        preview.buttonNext.click()                               # 2 rows: wraps to the first one
        self.assertEqual(preview.computer.fid, fid)
        self.assertEqual(preview.currentMethod(), BaseLevelOptions.MANUAL_BASE_LEVEL)
        self.assertAlmostEqual(preview.spinLevel.value(), 99.5)
        preview.buttonPrevious.click()
        self.assertEqual(preview.computer.fid, fid_b)
        self.assertEqual(len(self.plugin.previews), 1)          # still one window
        # a method chosen now applies to the polygon shown only
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.APPROXIMATE_VIA_AVG.code))
        self.assertEqual(dlg.tablePolygons.cellWidget(1, COL_METHOD).currentData(),
                         BaseLevelOptions.APPROXIMATE_VIA_AVG.code)
        self.assertEqual(dlg.tablePolygons.cellWidget(0, COL_METHOD).currentData(),
                         BaseLevelOptions.MANUAL_BASE_LEVEL.code)
        self.assertEqual(self.values("Base_meth"), {"pile A": "MANUAL", "pile B": "AVG"})
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.MANUAL_BASE_LEVEL.code))
        preview.spinLevel.setValue(100.25)
        preview.close()                              # level typed just before closing: kept too
        self.assertEqual(self.values("Base_meth")["pile B"], "MANUAL")
        self.assertAlmostEqual(self.values("Base_z")["pile B"], 100.25)
        self.plugin.dlg.close()                      # a new dialog reads the saved choice back
        self.plugin.dlg = None
        dlg = self.open_dialog()
        self.assertEqual(dlg.tablePolygons.cellWidget(1, COL_METHOD).currentData(),
                         BaseLevelOptions.MANUAL_BASE_LEVEL.code)
        self.assertAlmostEqual(dlg.tablePolygons.cellWidget(1, COL_LEVEL).value(), 100.25)

    def test_web_tiles_as_base_dem(self):
        """A web map (XYZ tiles) chosen as base DEM: the 3D view still opens, and the calculation
        refuses the 'Base DEM' method with a clear message instead of failing in the task."""
        from volume_calculation_tool.volume_calculation_tool_dialog import COL_ID
        tiles = QgsRasterLayer("type=xyz&url=http://127.0.0.1:1/%7Bz%7D/%7Bx%7D/%7By%7D.png&zmax=19&zmin=0",
                               "Google Satellite", "wms")
        self.assertTrue(tiles.isValid())
        QgsProject.instance().addMapLayer(tiles)
        dlg = self.open_dialog()
        dlg.mFieldComboHeightLayerBase.setLayer(tiles)
        self.set_row(dlg, 0, BaseLevelOptions.SPLINE_RING)
        fid = dlg.tablePolygons.item(0, COL_ID).data(Qt.ItemDataRole.UserRole)
        self.plugin.showBasePreview(fid)
        self.assertEqual(self.messages, [])
        preview = self.plugin.previews[0]
        self.assertIn("above the base", preview.labelInfo.text())
        preview.comboMethod.setCurrentIndex(preview.comboMethod.findData(BaseLevelOptions.USE_DEM_LAYER.code))
        self.assertIn("not a DEM with a pixel grid", preview.labelInfo.text())
        preview.close()

        self.set_row(dlg, 1, BaseLevelOptions.USE_DEM_LAYER)
        self.plugin.workflow()
        self.assertIsNone(self.plugin.task)
        self.assertTrue(any(m[0] == "warning" and "not a DEM with a pixel grid" in m[1] for m in self.messages),
                        self.messages)

    def test_geographic_dem_refused(self):
        path = os.path.join(TMP, "geo.tif")
        write_dem(path, np.full((SIZE, SIZE), 1.0), epsg=4326, pixel=0.0001)
        geo = QgsRasterLayer(path, "geo")
        QgsProject.instance().addMapLayer(geo)
        dlg = self.open_dialog()
        dlg.mFieldComboHeightLayer.setLayer(geo)
        self.plugin.workflow()
        self.assertIsNone(self.plugin.task)
        self.assertTrue(any("geographic" in m[1] for m in self.messages), self.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
