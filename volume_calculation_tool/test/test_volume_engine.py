# -*- coding: utf-8 -*-
"""Headless tests of the volume engine on synthetic DEMs with known volumes.

Run with the QGIS Python, e.g. on Windows:
    "C:\\Program Files\\QGIS 3.44.0\\bin\\python-qgis.bat" -m unittest discover -s volume_calculation_tool/test -v
(from the repository root)
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_PARENT = os.path.dirname(os.path.dirname(HERE))
if PLUGIN_PARENT not in sys.path:
    sys.path.insert(0, PLUGIN_PARENT)

from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsGeometry,  # noqa: E402
                       QgsProject, QgsRasterLayer)

if QgsApplication.instance() is None:  # may already exist when imported by another test
    _APP = QgsApplication([], False)
    _APP.initQgis()

from volume_calculation_tool.volume_engine import (  # noqa: E402
    AccurateVolumeCalculationTask, BaseLevelOptions, CalculationSettings,
    PolygonJob, crs_linear_factor_to_meters)

X0, Y0 = 500000.0, 5000050.0   # top-left corner of the DEMs
SIZE = 50                      # 50 x 50 pixels of 1 m


def write_dem(path, array, epsg=32632, pixel=1.0, nodata=-9999.0):
    rows, cols = array.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((X0, pixel, 0, Y0, 0, -pixel))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array.astype(np.float32))
    ds = None


def square(xmin_local, ymin_local, size):
    """Square polygon in local metres (origin = bottom-left of the DEM)."""
    x = X0 + xmin_local
    y = Y0 - SIZE + ymin_local
    return QgsGeometry.fromWkt(
        "POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))".format(
            x0=x, y0=y, x1=x + size, y1=y + size))


def local_coords():
    cols = np.arange(SIZE) + 0.5
    rows = np.arange(SIZE) + 0.5
    x = np.tile(cols, (SIZE, 1))
    y = np.tile((SIZE - rows)[:, None], (1, SIZE))  # local y of pixel centres
    return x, y


def box(z, x0, y0, size, height):
    """Add a box of `height` on pixels whose centre lies in [x0, x0+size] x [y0, y0+size]."""
    x, y = local_coords()
    z = z.copy()
    inside = (x > x0) & (x < x0 + size) & (y > y0) & (y < y0 + size)
    z[inside] += height
    return z


class EngineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="voltest_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def layer(self, name, array, **kw):
        path = os.path.join(self.tmp, name + ".tif")
        write_dem(path, array, **kw)
        lyr = QgsRasterLayer(path, name)
        self.assertTrue(lyr.isValid())
        return lyr

    def run_task(self, height_layer, jobs, step=None, base_layer=None, ndom_path=None, ring_width=3.0):
        ext = height_layer.extent()
        px = height_layer.rasterUnitsPerPixelX()
        py = height_layer.rasterUnitsPerPixelY()
        factor = crs_linear_factor_to_meters(height_layer.crs())
        settings = CalculationSettings(
            height_layer.dataProvider().clone(), 1, ext.xMinimum(), ext.yMaximum(), px, py,
            step or px, step or py, factor,
            base_provider=base_layer.dataProvider().clone() if base_layer else None,
            ndom_path=ndom_path, ndom_crs_wkt=height_layer.crs().toWkt(), ring_width=ring_width)
        task = AccurateVolumeCalculationTask("test", jobs, settings)
        self.assertTrue(task.run(), task.exception_trace)
        return task

    # ------------------------------------------------------------------
    def test_box_on_flat_ground_all_constant_methods(self):
        ground = np.full((SIZE, SIZE), 100.0)
        dem = self.layer("flat_box", box(ground, 20, 20, 10, 5.0))
        poly = square(15.5, 15.5, 19)  # outline on pixel centres, box fully inside
        jobs = [PolygonJob(1, "min", poly, BaseLevelOptions.APPROXIMATE_VIA_MIN),
                PolygonJob(2, "avg", poly, BaseLevelOptions.APPROXIMATE_VIA_AVG),
                PolygonJob(3, "plane", poly, BaseLevelOptions.BEST_FIT_PLANE),
                PolygonJob(4, "manual", poly, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)]
        task = self.run_task(dem, jobs)
        for fid in (1, 2, 3, 4):
            res = task.results[fid]
            self.assertIsNone(res.error)
            self.assertAlmostEqual(res.above_m3, 500.0, places=4, msg=str(fid))
            self.assertAlmostEqual(res.below_m3, 0.0, places=4)
            self.assertAlmostEqual(res.coverage, 1.0)

    def test_sloped_ground_plane_is_exact_min_overestimates(self):
        x, _ = local_coords()
        ground = 100.0 + 0.1 * x
        dem = self.layer("slope_box", box(ground, 20, 20, 10, 5.0))
        poly = square(15.5, 15.5, 19)
        jobs = [PolygonJob(1, "plane", poly, BaseLevelOptions.BEST_FIT_PLANE),
                PolygonJob(2, "min", poly, BaseLevelOptions.APPROXIMATE_VIA_MIN)]
        task = self.run_task(dem, jobs)
        # outline heights are read pixel-wise (stepped), so the fit is close but not exact
        self.assertAlmostEqual(task.results[1].plane[0], 0.1, delta=0.001)
        self.assertAlmostEqual(task.results[1].above_m3, 500.0, delta=1.0)
        self.assertGreater(task.results[2].above_m3, 600.0)

    def test_tin_and_spline_on_sloped_ground(self):
        x, _ = local_coords()
        dem = self.layer("slope_tin", box(100.0 + 0.1 * x, 20, 20, 10, 5.0))
        # vertices on pixel centres: a vertex on a pixel edge takes the next pixel (+half a pixel of slope)
        poly = square(15.5, 15.5, 19)
        jobs = [PolygonJob(1, "tin", poly, BaseLevelOptions.TRIANGULATED_OUTLINE),
                PolygonJob(2, "spline", poly, BaseLevelOptions.SPLINE_RING)]
        task = self.run_task(dem, jobs)
        for fid, delta in ((1, 0.01), (2, 5.0)):
            res = task.results[fid]
            self.assertIsNone(res.error, res.error)
            self.assertAlmostEqual(res.above_m3, 500.0, delta=delta, msg=res.method)

    def test_spline_ring_and_stiffness_per_polygon(self):
        """Stiffness 4 = the former fixed spline; 10 = nearly the plane of the ring (cannot follow the
        valley); the ring width of each job is used instead of the global one."""
        from volume_calculation_tool.volume_engine import SPLINE_SMOOTHING, spline_smoothing
        self.assertAlmostEqual(spline_smoothing(4.0), SPLINE_SMOOTHING)
        x, _ = local_coords()
        ground = 100.0 + 0.01 * (x - 25.0) ** 2            # valley across the polygon
        dem = self.layer("valley_stiffness", box(ground, 20, 20, 10, 5.0))
        poly = square(15, 15, 20)
        jobs = [PolygonJob(1, "default", poly, BaseLevelOptions.SPLINE_RING),
                PolygonJob(2, "stiff 4", poly, BaseLevelOptions.SPLINE_RING, ring_width=5.0, stiffness=4.0),
                PolygonJob(3, "stiff 10", poly, BaseLevelOptions.SPLINE_RING, ring_width=5.0, stiffness=10.0),
                PolygonJob(4, "ring 2", poly, BaseLevelOptions.SPLINE_RING, ring_width=2.0)]
        task = self.run_task(dem, jobs, ring_width=5.0)
        res = task.results
        self.assertEqual((res[1].ring_width, res[1].stiffness), (5.0, 4.0))
        self.assertAlmostEqual(res[1].above_m3, res[2].above_m3, places=9)   # defaults = 5 m / 4
        self.assertEqual(res[4].ring_width, 2.0)
        self.assertNotAlmostEqual(res[4].above_m3, res[2].above_m3, places=3)
        self.assertLess(abs(res[2].above_m3 - 500.0), 10.0)                  # follows the valley
        self.assertGreater(abs(res[3].above_m3 - 500.0), 3 * abs(res[2].above_m3 - 500.0))

    def test_tin_uses_only_the_drawn_vertices(self):
        """As Pix4D 'Triangulated': a bank crossed by an edge between two vertices (placed on the
        natural ground) must not lift the base. Dense outline sampling gave ~490 m3 here."""
        ground = np.full((SIZE, SIZE), 100.0)
        z = box(ground, 20, 20, 10, 5.0)       # pile: 500 m3
        z = box(z, 23, 33, 4, 2.0)              # 4 x 4 m bank across the top edge (y = 35)
        dem = self.layer("tin_vertices", z)
        poly = square(15, 15, 20)               # vertices at the 4 corners, on flat ground at 100 m
        task = self.run_task(dem, [PolygonJob(1, "tin", poly, BaseLevelOptions.TRIANGULATED_OUTLINE)])
        res = task.results[1]
        self.assertIsNone(res.error, res.error)
        self.assertEqual(res.fit_points, 4)
        # base flat at 100 m: pile + the 2 rows of the bank inside the polygon (8 cells x 2 m)
        self.assertAlmostEqual(res.above_m3, 516.0, places=6)
        self.assertAlmostEqual(res.below_m3, 0.0, places=6)

    def test_spline_follows_curved_valley(self):
        x, _ = local_coords()
        rng = np.random.default_rng(3)
        ground = 100.0 + 0.01 * (x - 25.0) ** 2 + rng.normal(0, 0.03, x.shape)  # valley + 3 cm noise
        dem = self.layer("valley", box(ground, 20, 20, 10, 5.0))
        poly = square(15, 15, 20)
        jobs = [PolygonJob(1, "plane", poly, BaseLevelOptions.BEST_FIT_PLANE),
                PolygonJob(2, "spline", poly, BaseLevelOptions.SPLINE_RING)]
        task = self.run_task(dem, jobs, ring_width=5.0)
        plane, spline = task.results[1], task.results[2]
        self.assertIsNone(spline.error, spline.error)
        # with a perfect base, the 3 cm noise of the 300 ground cells alone adds ~ +3.6 m3 "above"
        self.assertAlmostEqual(spline.above_m3, 503.6, delta=5.0)
        self.assertGreater(abs(plane.above_m3 - 500.0), 5 * abs(spline.above_m3 - 503.6))
        self.assertLess(spline.fit_rms, 0.06)

    def test_spline_ring_excludes_neighbour_polygon(self):
        ground = np.full((SIZE, SIZE), 100.0)
        z = box(ground, 20, 20, 10, 5.0)       # pile A
        z = box(z, 37, 20, 10, 8.0)            # pile B, 2 m away from polygon A
        dem = self.layer("neighbours", z)
        jobs = [PolygonJob(1, "A", square(15, 15, 20), BaseLevelOptions.SPLINE_RING),
                PolygonJob(2, "B", square(35, 15, 16), BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)]
        task = self.run_task(dem, jobs, ring_width=5.0)
        self.assertAlmostEqual(task.results[1].above_m3, 500.0, delta=2.0)

    def test_surface_to_surface(self):
        x, _ = local_coords()
        ground = 100.0 + 0.1 * x
        base = self.layer("s2s_base", ground)
        top = self.layer("s2s_top", box(box(ground, 5, 5, 10, 5.0), 30, 30, 10, -2.0))
        poly = square(0, 0, 50)
        task = self.run_task(top, [PolygonJob(1, "all", poly, BaseLevelOptions.USE_DEM_LAYER)],
                             base_layer=base)
        res = task.results[1]
        self.assertAlmostEqual(res.above_m3, 500.0, places=3)
        self.assertAlmostEqual(res.below_m3, 200.0, places=3)

    def test_mixed_methods_in_one_run(self):
        ground = np.full((SIZE, SIZE), 100.0)
        z = box(ground, 5, 5, 10, 5.0)        # stockpile, 500 m3 above 100 m
        z = box(z, 30, 30, 10, -3.0)           # pit, 300 m3 below 100 m
        dem = self.layer("mixed", z)
        jobs = [PolygonJob(1, "pile", square(2.5, 2.5, 15), BaseLevelOptions.APPROXIMATE_VIA_MIN),
                PolygonJob(2, "pit", square(27, 27, 16), BaseLevelOptions.MANUAL_BASE_LEVEL, 99.0)]
        task = self.run_task(dem, jobs)
        self.assertAlmostEqual(task.results[1].above_m3, 500.0, places=4)
        # pit with manual base 99: ground 1 m above over 16x16=256 cells minus the 100 pit cells
        res = task.results[2]
        self.assertAlmostEqual(res.above_m3, 256.0 - 100.0, places=4)
        self.assertAlmostEqual(res.below_m3, 100.0 * 2.0, places=4)

    def test_oversampling_gives_same_volume(self):
        ground = np.full((SIZE, SIZE), 100.0)
        dem = self.layer("oversample", box(ground, 20, 20, 10, 5.0))
        poly = square(15, 15, 20)  # edges on pixel borders
        task = self.run_task(dem, [PolygonJob(1, "p", poly, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)],
                             step=0.25)
        self.assertAlmostEqual(task.results[1].above_m3, 500.0, places=4)

    def test_nodata_reduces_coverage(self):
        ground = np.full((SIZE, SIZE), 100.0)
        z = box(ground, 20, 20, 10, 5.0)
        z[0:5, :] = -9999.0  # top 5 rows nodata
        dem = self.layer("nodata", z)
        poly = square(0, 0, 50)
        task = self.run_task(dem, [PolygonJob(1, "p", poly, BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)])
        res = task.results[1]
        self.assertAlmostEqual(res.coverage, 0.9)
        self.assertAlmostEqual(res.above_m3, 500.0, places=4)

    def test_outline_outside_dem_is_an_error_not_zero(self):
        ground = np.full((SIZE, SIZE), 100.0)
        dem = self.layer("outside", ground)
        poly = square(200, 200, 10)
        task = self.run_task(dem, [PolygonJob(1, "p", poly, BaseLevelOptions.APPROXIMATE_VIA_AVG)])
        self.assertIsNotNone(task.results[1].error)

    def test_feet_crs_converted_to_m3(self):
        ground = np.full((SIZE, SIZE), 100.0)
        dem = self.layer("feet", box(ground, 20, 20, 10, 5.0), epsg=2263)  # NY Long Island, US ft
        poly = square(15.5, 15.5, 19)
        task = self.run_task(dem, [PolygonJob(1, "p", poly, BaseLevelOptions.APPROXIMATE_VIA_MIN)])
        self.assertAlmostEqual(task.results[1].above_m3, 500.0 * 0.3048006096 ** 3, places=3)

    def test_ndom_written(self):
        ground = np.full((SIZE, SIZE), 100.0)
        dem = self.layer("ndom_src", box(ground, 20, 20, 10, 5.0))
        poly = square(15, 15, 20)  # edges on pixel borders -> 20 x 20 pixels
        out = os.path.join(self.tmp, "ndom.tif")
        task = self.run_task(dem, [PolygonJob(1, "p", poly, BaseLevelOptions.APPROXIMATE_VIA_MIN)],
                             step=2.0, ndom_path=out)
        self.assertTrue(task.ndom_written)
        self.assertAlmostEqual(task.ndom_abs_max, 5.0, places=4)
        ds = gdal.Open(out)
        arr = ds.GetRasterBand(1).ReadAsArray()
        valid = arr != -9999.0
        self.assertEqual(int(valid.sum()), 20 * 20)
        self.assertAlmostEqual(float(arr[valid].sum()), 500.0, places=3)
        ds = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
