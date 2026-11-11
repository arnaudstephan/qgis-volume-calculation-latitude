# -*- coding: utf-8 -*-
"""Independent verification of the volume calculations.

The engine is checked against things that do NOT share its code:

1. A reference implementation written here with numpy only (GDAL ReadAsArray, own
   even-odd point-in-polygon test, own nearest-neighbour sampling), on hundreds of
   randomised scenes: random DEM (large Lambert-93 coordinates, non-square pixels,
   nodata holes), random polygons (holes, multipart, partly outside the DEM), random
   sampling steps, manual / outline / second-DEM bases. Volumes AND cell counts must agree.
2. Closed-form volumes (cone, paraboloid, spherical cap, rotated frustum) and convergence.
3. QGIS's own algorithm (GDAL clip by mask + native "Raster surface volume").
4. Invariances: additivity of split polygons, vertex order, translation.
5. Units (international / US survey feet, ft3, yd3), scale/offset integer DEMs, band choice.
6. Base surfaces: empty ground must give ~0 m3 for plane / TIN / spline, ordering MIN >= AVG.
7. nDOM integral equals the volume.

Run with the QGIS Python from the repository root:
    "C:\\Program Files\\QGIS 3.44.0\\bin\\python-qgis.bat" volume_calculation_tool/test/test_volume_verification.py
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from qgis.core import QgsApplication, QgsGeometry, QgsRasterLayer, QgsVectorLayer  # noqa: E402

if QgsApplication.instance() is None:
    _APP = QgsApplication([], False)
    _APP.initQgis()

from volume_calculation_tool.volume_engine import (  # noqa: E402
    AccurateVolumeCalculationTask, BaseLevelOptions, CalculationSettings, PolygonJob,
    convert_volume, crs_linear_factor_to_meters)

TMP = tempfile.mkdtemp(prefix="volverif_")
NODATA = -9999.0


# ======================================================================
# Test data
# ======================================================================

def write_raster(path, bands, x0, y0, px, py, epsg=2154, dtype=gdal.GDT_Float32,
                 nodata=NODATA, scale=None, offset=None):
    """bands: list of 2D arrays (row 0 = north)."""
    rows, cols = bands[0].shape
    ds = gdal.GetDriverByName("GTiff").Create(path, cols, rows, len(bands), dtype)
    ds.SetGeoTransform((x0, px, 0.0, y0, 0.0, -py))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    for i, arr in enumerate(bands, start=1):
        band = ds.GetRasterBand(i)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if scale is not None:
            band.SetScale(scale)
        if offset is not None:
            band.SetOffset(offset)
        band.WriteArray(arr)
    ds = None
    return path


def star_ring(rng, cx, cy, rmin, rmax, n=None, clockwise=False):
    n = n or int(rng.integers(5, 40))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(rmin, rmax, n)
    ring = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
    return ring[::-1] if clockwise else ring


def to_wkt(polygons):
    def ring_txt(ring):
        pts = list(ring) + [ring[0]]
        return "(" + ",".join("{!r} {!r}".format(float(x), float(y)) for x, y in pts) + ")"
    parts = ["(" + ",".join(ring_txt(r) for r in rings) + ")" for rings in polygons]
    return "MULTIPOLYGON(" + ",".join(parts) + ")"


def to_geometry(polygons):
    geom = QgsGeometry.fromWkt(to_wkt(polygons))
    assert not geom.isNull(), "bad test polygon"
    return geom


# ======================================================================
# Reference implementation (does not use the engine, QGIS rasters or GDAL rasterize)
# ======================================================================

def read_band(path, band=1):
    ds = gdal.Open(path)
    b = ds.GetRasterBand(band)
    raw = b.ReadAsArray().astype(np.float64)
    nodata = b.GetNoDataValue()
    scale = b.GetScale() if b.GetScale() is not None else 1.0
    offset = b.GetOffset() if b.GetOffset() is not None else 0.0
    data = raw * scale + offset
    if nodata is not None:
        data[raw == nodata] = np.nan
    data[~np.isfinite(raw)] = np.nan
    gt = ds.GetGeoTransform()
    ds = None
    return data, gt


def sample_nearest(data, gt, xc, yc):
    """Value of the pixel containing each cell centre; xc: 1D (cols), yc: 1D (rows)."""
    col = np.floor((xc - gt[0]) / gt[1]).astype(np.int64)
    row = np.floor((yc - gt[3]) / gt[5]).astype(np.int64)
    out = np.full((yc.size, xc.size), np.nan)
    vc = (col >= 0) & (col < data.shape[1])
    vr = (row >= 0) & (row < data.shape[0])
    out[np.ix_(vr, vc)] = data[np.ix_(row[vr], col[vc])]
    return out


def inside_polygons(X, Y, polygons):
    """Even-odd ray casting per polygon (holes subtract), union over the parts."""
    result = np.zeros(X.shape, dtype=bool)
    for rings in polygons:
        inside = np.zeros(X.shape, dtype=bool)
        for ring in rings:
            x1, y1 = ring[:, 0], ring[:, 1]
            x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
            for a, b, c, d in zip(x1, y1, x2, y2):
                if b == d:
                    continue
                crosses = (b > Y) != (d > Y)
                x_cross = (c - a) * (Y - b) / (d - b) + a
                inside ^= crosses & (X < x_cross)
        result |= inside
    return result


def reference_volume(dem_path, polygons, step_x, step_y, base, band=1, factor=1.0):
    """base: float, or callable(X, Y) -> array, or ('dem', path, band).
    Returns (above_m3, below_m3, cells_inside, cells_valid)."""
    data, gt = read_band(dem_path, band)
    x0, y0 = gt[0], gt[3]
    pts = np.vstack([r for rings in polygons for r in rings])
    eps = 1e-9
    c0 = math.floor((pts[:, 0].min() - x0) / step_x + eps)
    c1 = math.ceil((pts[:, 0].max() - x0) / step_x - eps)
    r0 = math.floor((y0 - pts[:, 1].max()) / step_y + eps)
    r1 = math.ceil((y0 - pts[:, 1].min()) / step_y - eps)
    xc = x0 + (np.arange(c0, max(c1, c0 + 1)) + 0.5) * step_x
    yc = y0 - (np.arange(r0, max(r1, r0 + 1)) + 0.5) * step_y
    X, Y = np.meshgrid(xc, yc)
    inside = inside_polygons(X, Y, polygons)
    height = sample_nearest(data, gt, xc, yc)
    if isinstance(base, tuple):
        bdata, bgt = read_band(base[1], base[2])
        base_grid = sample_nearest(bdata, bgt, xc, yc)
    elif callable(base):
        base_grid = base(X, Y)
    else:
        base_grid = np.full(X.shape, float(base))
    dz = height - base_grid
    valid = inside & np.isfinite(dz)
    d = dz[valid]
    unit = step_x * step_y * factor ** 3
    return (float(d[d > 0].sum()) * unit, float(-d[d < 0].sum()) * unit,
            int(inside.sum()), int(valid.sum()))


# ======================================================================
# Engine runner
# ======================================================================

def run_engine(dem_path, jobs, step_x=None, step_y=None, band=1, base_path=None, base_band=1,
               ring_width=3.0, ndom_path=None):
    layer = QgsRasterLayer(dem_path, "dem")
    assert layer.isValid()
    ext = layer.extent()
    px = layer.rasterUnitsPerPixelX()
    py = layer.rasterUnitsPerPixelY()
    base_provider = None
    if base_path:
        base_layer = QgsRasterLayer(base_path, "base")
        assert base_layer.isValid()
        base_provider = base_layer.dataProvider().clone()
    settings = CalculationSettings(
        layer.dataProvider().clone(), band, ext.xMinimum(), ext.yMaximum(), px, py,
        step_x or px, step_y or py, crs_linear_factor_to_meters(layer.crs()),
        base_provider=base_provider, base_band=base_band, ring_width=ring_width,
        ndom_path=ndom_path, ndom_crs_wkt=layer.crs().toWkt() if ndom_path else "")
    task = AccurateVolumeCalculationTask("verification", jobs, settings)
    assert task.run(), task.exception_trace
    return task


# ======================================================================
# Random scenes
# ======================================================================

def random_terrain(rng, nrows, ncols, px, py, x0, y0):
    xs = x0 + (np.arange(ncols) + 0.5) * px
    ys = y0 - (np.arange(nrows) + 0.5) * py
    X, Y = np.meshgrid(xs, ys)
    u = (X - X.mean()) / (ncols * px)
    v = (Y - Y.mean()) / (nrows * py)
    z = rng.uniform(50, 1500) + rng.normal(0, 3) * u + rng.normal(0, 3) * v
    for _ in range(int(rng.integers(1, 6))):
        cx, cy = rng.uniform(-0.4, 0.4, 2)
        width = rng.uniform(0.05, 0.3)
        z = z + rng.uniform(-8, 12) * np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2 * width ** 2))
    z = z + rng.normal(0, 0.05, z.shape)
    if rng.random() < 0.5:  # nodata holes
        for _ in range(int(rng.integers(1, 4))):
            r, c = rng.integers(0, nrows), rng.integers(0, ncols)
            z[max(0, r - 6):r + 6, max(0, c - 6):c + 6] = NODATA
    return z.astype(np.float32)


def random_polygons(rng, x0, y0, width, height):
    """Random valid (multi)polygon. Star rings with few vertices can self-intersect
    (angular gap > 180 deg): such draws are rejected, as the plugin repairs invalid
    geometries before computing anyway."""
    for _ in range(100):
        polygons = _random_polygons(rng, x0, y0, width, height)
        if to_geometry(polygons).isGeosValid():
            return polygons
    raise RuntimeError("could not draw a valid polygon")


def _random_polygons(rng, x0, y0, width, height):
    polygons = []
    for _ in range(1 if rng.random() < 0.7 else 2):
        # centre may be close to the border so that the polygon goes outside the DEM
        cx = x0 + rng.uniform(0.1, 0.95) * width
        cy = y0 - rng.uniform(0.1, 0.95) * height
        rmax = rng.uniform(0.08, 0.3) * min(width, height)
        rings = [star_ring(rng, cx, cy, 0.5 * rmax, rmax)]
        if rng.random() < 0.3:  # hole
            rings.append(star_ring(rng, cx, cy, 0.1 * rmax, 0.3 * rmax, clockwise=True))
        polygons.append(rings)
    if len(polygons) == 2:
        g = [to_geometry([p]) for p in polygons]
        if g[0].intersects(g[1]):   # parts of a multipolygon must not overlap
            polygons = polygons[:1]
    return polygons


class Scene:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.px = float(rng.choice([0.1, 0.25, 0.37, 0.5, 1.0]))
        self.py = self.px if rng.random() < 0.7 else float(rng.choice([0.2, 0.5, 0.8]))
        self.ncols = int(rng.integers(60, 180))
        self.nrows = int(rng.integers(60, 180))
        self.x0 = float(rng.uniform(100000, 1200000)) + 0.123
        self.y0 = float(rng.uniform(6000000, 7100000)) + 0.456   # Lambert-93 like
        self.z = random_terrain(rng, self.nrows, self.ncols, self.px, self.py, self.x0, self.y0)
        self.path = write_raster(os.path.join(TMP, "scene_{}.tif".format(seed)), [self.z],
                                 self.x0, self.y0, self.px, self.py)
        self.width = self.ncols * self.px
        self.height = self.nrows * self.py
        self.polygons = random_polygons(rng, self.x0, self.y0, self.width, self.height)
        self.geometry = to_geometry(self.polygons)
        choice = rng.random()
        if choice < 0.5:
            self.step_x, self.step_y = self.px, self.py
        else:
            self.step_x = self.px * float(rng.uniform(0.3, 2.5))
            self.step_y = self.py * float(rng.uniform(0.3, 2.5))
        valid = self.z[self.z != NODATA]
        self.level = float(rng.uniform(valid.min(), valid.max()))


def assert_close(test, engine_value, reference_value, what):
    tolerance = 1e-9 * max(1.0, abs(reference_value)) + 1e-6
    test.assertLessEqual(abs(engine_value - reference_value), tolerance,
                         "{}: engine {!r} != reference {!r}".format(what, engine_value, reference_value))


# ======================================================================
# Tests
# ======================================================================

class ReferenceComparisonTest(unittest.TestCase):
    """Engine == independent reference, on many random scenes."""

    N_SCENES = 150

    def test_random_scenes_manual_base(self):
        for seed in range(self.N_SCENES):
            with self.subTest(seed=seed):
                s = Scene(seed)
                task = run_engine(s.path, [PolygonJob(1, "p", s.geometry, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level)],
                                  s.step_x, s.step_y)
                res = task.results[1]
                ref = reference_volume(s.path, s.polygons, s.step_x, s.step_y, s.level)
                self.assertIsNone(res.error)
                self.assertEqual(res.cells_inside, ref[2], "cells inside")
                self.assertEqual(res.cells_valid, ref[3], "valid cells")
                assert_close(self, res.above_m3, ref[0], "above")
                assert_close(self, res.below_m3, ref[1], "below")

    def test_random_scenes_outline_bases(self):
        """MIN / AVG / PLANE: the volume integrated with the base the engine reports."""
        for seed in range(1000, 1000 + 60):
            s = Scene(seed)
            for method in (BaseLevelOptions.APPROXIMATE_VIA_MIN, BaseLevelOptions.APPROXIMATE_VIA_AVG,
                           BaseLevelOptions.BEST_FIT_PLANE):
                with self.subTest(seed=seed, method=method.code):
                    res = run_engine(s.path, [PolygonJob(1, "p", s.geometry, method)], s.step_x, s.step_y).results[1]
                    if res.error:   # outline entirely on nodata / outside: must be reported, never 0
                        self.assertIn("outline", res.error)
                        continue
                    if res.plane is not None:
                        a, b, c, xc, yc = res.plane
                        base = (lambda X, Y, a=a, b=b, c=c, xc=xc, yc=yc: a * (X - xc) + b * (Y - yc) + c)
                    else:
                        base = res.base_z
                    ref = reference_volume(s.path, s.polygons, s.step_x, s.step_y, base)
                    assert_close(self, res.above_m3, ref[0], "above")
                    assert_close(self, res.below_m3, ref[1], "below")

    def test_random_scenes_second_dem(self):
        """Surface to surface with a base DEM of another resolution and origin."""
        for seed in range(2000, 2000 + 40):
            with self.subTest(seed=seed):
                s = Scene(seed)
                rng = np.random.default_rng(seed + 7)
                bpx = float(rng.choice([0.3, 0.5, 1.0, 2.0]))
                bx0 = s.x0 - float(rng.uniform(0, 3))
                by0 = s.y0 + float(rng.uniform(0, 3))
                bcols = int((s.width + 6) / bpx) + 1
                brows = int((s.height + 6) / bpx) + 1
                base_z = random_terrain(rng, brows, bcols, bpx, bpx, bx0, by0)
                base_path = write_raster(os.path.join(TMP, "base_{}.tif".format(seed)), [base_z], bx0, by0, bpx, bpx)
                res = run_engine(s.path, [PolygonJob(1, "p", s.geometry, BaseLevelOptions.USE_DEM_LAYER)],
                                 s.step_x, s.step_y, base_path=base_path).results[1]
                ref = reference_volume(s.path, s.polygons, s.step_x, s.step_y, ("dem", base_path, 1))
                self.assertEqual(res.cells_valid, ref[3], "valid cells")
                assert_close(self, res.above_m3, ref[0], "above")
                assert_close(self, res.below_m3, ref[1], "below")


class ReprojectedBaseDemTest(unittest.TestCase):
    """Base DEM in another CRS: the reference transforms every cell centre with QGIS
    (QgsCoordinateTransform), the engine with GDAL/PROJ."""

    def test_base_dem_in_another_crs(self):
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsPointXY, QgsProject
        for seed in range(7000, 7010):
            with self.subTest(seed=seed):
                s = Scene(seed)
                rng = np.random.default_rng(seed)
                # base terrain built in Lambert-93 then warped to UTM 31N (EPSG:32631)
                base_l93 = write_raster(os.path.join(TMP, "b93_{}.tif".format(seed)),
                                        [random_terrain(rng, s.nrows, s.ncols, s.px, s.py, s.x0, s.y0)],
                                        s.x0, s.y0, s.px, s.py)
                base_utm = os.path.join(TMP, "butm_{}.tif".format(seed))
                gdal.Warp(base_utm, base_l93, dstSRS="EPSG:32631", resampleAlg="near", dstNodata=NODATA)
                dem_wkt = QgsRasterLayer(s.path, "d").crs().toWkt()
                base_wkt = QgsRasterLayer(base_utm, "b").crs().toWkt()

                layer = QgsRasterLayer(s.path, "dem")
                base_layer = QgsRasterLayer(base_utm, "base")
                ext = layer.extent()
                settings = CalculationSettings(
                    layer.dataProvider().clone(), 1, ext.xMinimum(), ext.yMaximum(), s.px, s.py, s.step_x, s.step_y,
                    1.0, base_provider=base_layer.dataProvider().clone(), base_crs_wkts=(dem_wkt, base_wkt))
                task = AccurateVolumeCalculationTask("v", [PolygonJob(1, "p", s.geometry, BaseLevelOptions.USE_DEM_LAYER)],
                                                     settings)
                self.assertTrue(task.run(), task.exception_trace)
                res = task.results[1]

                transform = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:2154"),
                                                   QgsCoordinateReferenceSystem("EPSG:32631"), QgsProject.instance())
                bdata, bgt = read_band(base_utm)

                def base_fn(X, Y):
                    out = np.full(X.shape, np.nan)
                    for idx in np.ndindex(X.shape):
                        p = transform.transform(QgsPointXY(float(X[idx]), float(Y[idx])))
                        col = math.floor((p.x() - bgt[0]) / bgt[1])
                        row = math.floor((p.y() - bgt[3]) / bgt[5])
                        if 0 <= row < bdata.shape[0] and 0 <= col < bdata.shape[1]:
                            out[idx] = bdata[row, col]
                    return out
                ref = reference_volume(s.path, s.polygons, s.step_x, s.step_y, base_fn)
                # the two PROJ pipelines may differ by far less than a millimetre: allow a
                # handful of cells to pick the neighbouring base pixel
                self.assertLessEqual(abs(res.cells_valid - ref[3]), 2)
                total = max(1.0, ref[0] + ref[1])
                self.assertLess(abs(res.above_m3 - ref[0]) / total, 1e-3)
                self.assertLess(abs(res.below_m3 - ref[1]) / total, 1e-3)


class AnalyticVolumeTest(unittest.TestCase):
    """Closed-form volumes of solids standing on flat ground (midpoint rule on the DEM)."""

    GROUND = 250.0

    def solid_dem(self, name, height_fn, half_size, pixel):
        n = int(round(2 * half_size / pixel))
        cx, cy = 650000.0, 6860000.0          # centre of the solid (Lambert-93)
        left, top = cx - half_size, cy + half_size
        xs = left + (np.arange(n) + 0.5) * pixel
        ys = top - (np.arange(n) + 0.5) * pixel
        X, Y = np.meshgrid(xs - cx, ys - cy)  # local coordinates centred on the solid
        z = self.GROUND + height_fn(X, Y)
        path = write_raster(os.path.join(TMP, "{}_{}.tif".format(name, pixel)), [z.astype(np.float64)],
                            left, top, pixel, pixel, dtype=gdal.GDT_Float64)
        return path, (cx, cy)

    def circle(self, centre, radius, n=256):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [[np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])]]

    def check(self, name, height_fn, exact, radius, tolerance_fine):
        errors = []
        for pixel in (0.5, 0.2, 0.1):
            path, centre = self.solid_dem(name, height_fn, radius + 5, pixel)
            geom = to_geometry(self.circle(centre, radius + 3))
            for method in (BaseLevelOptions.MANUAL_BASE_LEVEL, BaseLevelOptions.APPROXIMATE_VIA_MIN,
                           BaseLevelOptions.BEST_FIT_PLANE, BaseLevelOptions.TRIANGULATED_OUTLINE,
                           BaseLevelOptions.SPLINE_RING):
                res = run_engine(path, [PolygonJob(1, name, geom, method, self.GROUND)]).results[1]
                self.assertIsNone(res.error, res.error)
                error = abs(res.above_m3 - exact) / exact
                if method == BaseLevelOptions.MANUAL_BASE_LEVEL:
                    errors.append(error)
                self.assertLess(error, 0.02 if pixel == 0.5 else tolerance_fine * 2,
                                "{} {} pixel {}: {} vs {}".format(name, method.code, pixel, res.above_m3, exact))
        self.assertLess(errors[-1], tolerance_fine, "{}: relative error {}".format(name, errors))
        self.assertLessEqual(errors[-1], errors[0] + 1e-6, "{} does not converge: {}".format(name, errors))

    def test_cone(self):
        R, h = 10.0, 6.0
        self.check("cone", lambda X, Y: np.clip(h * (1 - np.hypot(X, Y) / R), 0, None),
                   math.pi * R ** 2 * h / 3, R, 0.002)

    def test_paraboloid(self):
        R, h = 12.0, 5.0
        self.check("paraboloid", lambda X, Y: np.clip(h * (1 - (X ** 2 + Y ** 2) / R ** 2), 0, None),
                   math.pi * R ** 2 * h / 2, R, 0.002)

    def test_spherical_cap(self):
        Rs, hc = 15.0, 4.0
        a = math.sqrt(hc * (2 * Rs - hc))

        def cap(X, Y):
            r2 = X ** 2 + Y ** 2
            return np.where(r2 < a * a, np.sqrt(np.clip(Rs ** 2 - r2, 0, None)) - (Rs - hc), 0.0)
        self.check("cap", cap, math.pi * hc ** 2 * (3 * Rs - hc) / 3, a, 0.002)

    def test_rotated_frustum(self):
        a1, a2, h, angle = 16.0, 6.0, 4.0, math.radians(27)

        def frustum(X, Y):
            u = X * math.cos(angle) + Y * math.sin(angle)
            v = -X * math.sin(angle) + Y * math.cos(angle)
            d = np.maximum(np.abs(u), np.abs(v))
            return np.clip(h * (a1 / 2 - d) / ((a1 - a2) / 2), 0, h)
        exact = h / 3 * (a1 ** 2 + a2 ** 2 + a1 * a2)
        self.check("frustum", frustum, exact, a1 / 2 * math.sqrt(2), 0.003)


class QgisNativeCrossCheckTest(unittest.TestCase):
    """Same numbers as QGIS's own tools: GDAL clip by mask + native:rastersurfacevolume."""

    @classmethod
    def setUpClass(cls):
        plugins = os.path.join(QgsApplication.prefixPath(), "python", "plugins")
        if plugins not in sys.path:
            sys.path.append(plugins)
        from processing.core.Processing import Processing
        Processing.initialize()
        import processing
        cls.processing = processing

    def test_against_raster_surface_volume(self):
        for seed in range(3000, 3012):
            with self.subTest(seed=seed):
                s = Scene(seed)
                gpkg = os.path.join(TMP, "mask_{}.gpkg".format(seed))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(2154)
                ds = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg)
                lyr = ds.CreateLayer("mask", srs, ogr.wkbMultiPolygon)
                feat = ogr.Feature(lyr.GetLayerDefn())
                feat.SetGeometry(ogr.CreateGeometryFromWkt(to_wkt(s.polygons)))
                lyr.CreateFeature(feat)
                ds = None
                clipped = os.path.join(TMP, "clip_{}.tif".format(seed))
                self.processing.run("gdal:cliprasterbymasklayer", {
                    "INPUT": s.path, "MASK": gpkg, "NODATA": NODATA, "CROP_TO_CUTLINE": False,
                    "KEEP_RESOLUTION": True, "OUTPUT": clipped})
                native = {}
                for method_index, key in ((0, "above"), (1, "below")):
                    out = self.processing.run("native:rastersurfacevolume", {
                        "INPUT": clipped, "BAND": 1, "LEVEL": s.level, "METHOD": method_index,
                        "OUTPUT_TABLE": "TEMPORARY_OUTPUT"})
                    native[key] = abs(out["VOLUME"])
                res = run_engine(s.path, [PolygonJob(1, "p", s.geometry, BaseLevelOptions.MANUAL_BASE_LEVEL,
                                                     s.level)]).results[1]
                for key, value in (("above", res.above_m3), ("below", res.below_m3)):
                    self.assertLessEqual(abs(value - native[key]), 1e-6 * max(1.0, native[key]) + 1e-4,
                                         "{}: engine {} vs QGIS {}".format(key, value, native[key]))


class InvarianceTest(unittest.TestCase):

    def test_additivity_of_split_polygons(self):
        for seed in range(4000, 4030):
            with self.subTest(seed=seed):
                s = Scene(seed)
                bbox = s.geometry.boundingBox()
                cut = bbox.xMinimum() + bbox.width() * 0.4137 + 1e-4
                left = QgsGeometry.fromRect(bbox.buffered(1))
                left = left.intersection(QgsGeometry.fromWkt("POLYGON(({a} {b},{c} {b},{c} {d},{a} {d},{a} {b}))".format(
                    a=bbox.xMinimum() - 1, b=bbox.yMinimum() - 1, c=cut, d=bbox.yMaximum() + 1)))
                part_left = s.geometry.intersection(left)
                part_right = s.geometry.difference(left)
                jobs = [PolygonJob(1, "whole", s.geometry, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level),
                        PolygonJob(2, "left", part_left, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level),
                        PolygonJob(3, "right", part_right, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level)]
                r = run_engine(s.path, jobs, s.step_x, s.step_y).results
                self.assertEqual(r[1].cells_inside, r[2].cells_inside + r[3].cells_inside)
                assert_close(self, r[1].above_m3, r[2].above_m3 + r[3].above_m3, "above")
                assert_close(self, r[1].below_m3, r[2].below_m3 + r[3].below_m3, "below")

    def test_vertex_order_and_start_vertex(self):
        for seed in range(4100, 4120):
            with self.subTest(seed=seed):
                s = Scene(seed)
                reordered = [[np.roll(r[::-1], 3, axis=0) for r in rings] for rings in s.polygons]
                jobs = [PolygonJob(1, "a", s.geometry, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level),
                        PolygonJob(2, "b", to_geometry(reordered), BaseLevelOptions.MANUAL_BASE_LEVEL, s.level)]
                r = run_engine(s.path, jobs, s.step_x, s.step_y).results
                self.assertEqual(r[1].cells_inside, r[2].cells_inside)
                assert_close(self, r[1].above_m3, r[2].above_m3, "above")

    def test_translation_small_vs_large_coordinates(self):
        rng = np.random.default_rng(5)
        z = random_terrain(rng, 120, 120, 0.25, 0.25, 0.0, 30.0)
        polygons = [[star_ring(rng, 15.0, 15.0, 5.0, 12.0, n=25)]]
        results = []
        for dx, dy in ((0.0, 0.0), (950000.0, 7000000.0)):
            path = write_raster(os.path.join(TMP, "shift_{}.tif".format(int(dx))), [z], dx, 30.0 + dy, 0.25, 0.25)
            shifted = [[r + np.array([dx, dy]) for r in rings] for rings in polygons]
            res = run_engine(path, [PolygonJob(1, "p", to_geometry(shifted), BaseLevelOptions.MANUAL_BASE_LEVEL,
                                               float(np.median(z)))]).results[1]
            results.append(res)
        self.assertEqual(results[0].cells_inside, results[1].cells_inside)
        assert_close(self, results[0].above_m3, results[1].above_m3, "above")


class UnitsAndFormatsTest(unittest.TestCase):

    def box_dem(self, name, epsg, dtype=gdal.GDT_Float32, scale=None, offset=None, raw=None, bands=None):
        z = np.full((40, 40), 100.0)
        z[10:30, 10:30] += 5.0  # 20 x 20 pixels x 5 units
        data = raw if raw is not None else z
        return write_raster(os.path.join(TMP, name + ".tif"), bands or [data], 1000.0, 1040.0, 1.0, 1.0,
                            epsg=epsg, dtype=dtype, scale=scale, offset=offset)

    def square(self):
        return to_geometry([[np.array([[1005.0, 1005.0], [1035.0, 1005.0], [1035.0, 1035.0], [1005.0, 1035.0]])]])

    def volume(self, path, band=1):
        return run_engine(path, [PolygonJob(1, "p", self.square(), BaseLevelOptions.MANUAL_BASE_LEVEL, 100.0)],
                          band=band).results[1].above_m3

    def test_metres(self):
        self.assertAlmostEqual(self.volume(self.box_dem("m", 2154)), 2000.0, places=9)

    def test_international_feet(self):
        self.assertAlmostEqual(self.volume(self.box_dem("ft", 2256)), 2000.0 * 0.3048 ** 3, places=9)

    def test_us_survey_feet(self):
        self.assertAlmostEqual(self.volume(self.box_dem("ftus", 2263)), 2000.0 * (1200.0 / 3937.0) ** 3, places=9)

    def test_output_unit_conversions(self):
        self.assertAlmostEqual(convert_volume(1.0, "m3"), 1.0, places=12)
        self.assertAlmostEqual(convert_volume(1.0, "ft3"), 35.31466672148859, places=9)
        self.assertAlmostEqual(convert_volume(1.0, "yd3"), 1.3079506193143922, places=9)
        self.assertAlmostEqual(convert_volume(convert_volume(764.554857984, "yd3"), "m3"), 1000.0, places=9)

    def test_integer_dem_with_scale_and_offset(self):
        z = np.full((40, 40), 100.0)
        z[10:30, 10:30] += 5.0
        raw = np.round((z - 90.0) / 0.01).astype(np.int16)  # stored in cm above 90 m
        path = self.box_dem("scaled", 2154, dtype=gdal.GDT_Int16, scale=0.01, offset=90.0, raw=raw)
        self.assertAlmostEqual(self.volume(path), 2000.0, places=6)

    def test_band_selection(self):
        z = np.full((40, 40), 100.0)
        z[10:30, 10:30] += 5.0
        junk = np.full((40, 40), 500.0)
        path = self.box_dem("bands", 2154, bands=[junk, z])
        self.assertAlmostEqual(self.volume(path, band=2), 2000.0, places=9)

    def test_geographic_crs_is_refused_by_the_unit_helper(self):
        from qgis.core import QgsCoordinateReferenceSystem
        self.assertIsNone(crs_linear_factor_to_meters(QgsCoordinateReferenceSystem("EPSG:4326")))


class BaseSurfaceTest(unittest.TestCase):
    """Without any stockpile the base methods must give (almost) no volume."""

    def ground_dem(self, name, fn, size=60, pixel=0.25):
        n = int(size / pixel)
        x0, y0 = 700000.0, 6600000.0 + size
        xs = (np.arange(n) + 0.5) * pixel
        X, Y = np.meshgrid(xs, size - xs)
        return write_raster(os.path.join(TMP, name + ".tif"), [fn(X, Y).astype(np.float64)], x0, y0, pixel, pixel,
                            dtype=gdal.GDT_Float64), (x0, y0 - size)

    def polygon(self, origin, rng):
        return to_geometry([[star_ring(rng, origin[0] + 30, origin[1] + 30, 8, 18, n=30)]])

    def test_empty_ground(self):
        rng = np.random.default_rng(11)
        terrains = {
            "flat": (lambda X, Y: 120.0 + 0 * X, [BaseLevelOptions.APPROXIMATE_VIA_MIN, BaseLevelOptions.APPROXIMATE_VIA_AVG]),
            "slope": (lambda X, Y: 120.0 + 0.08 * X - 0.05 * Y, []),
            "valley": (lambda X, Y: 120.0 + 0.004 * (X - 30) ** 2 + 0.02 * Y, []),
        }
        always = [BaseLevelOptions.BEST_FIT_PLANE, BaseLevelOptions.TRIANGULATED_OUTLINE, BaseLevelOptions.SPLINE_RING]
        for name, (fn, extra) in terrains.items():
            path, origin = self.ground_dem(name, fn)
            geom = self.polygon(origin, rng)
            area = geom.area()
            for method in extra + always:
                if name == "valley" and method in (BaseLevelOptions.BEST_FIT_PLANE, BaseLevelOptions.TRIANGULATED_OUTLINE):
                    continue  # a plane / linear TIN cannot follow the curvature: not expected to be ~0
                with self.subTest(terrain=name, method=method.code):
                    res = run_engine(path, [PolygonJob(1, name, geom, method)], ring_width=5.0).results[1]
                    self.assertIsNone(res.error, res.error)
                    mean_abs_dz = (res.above_m3 + res.below_m3) / area
                    self.assertLess(mean_abs_dz, 0.01, "{} {}: mean |dz| {:.4f} m".format(name, method.code, mean_abs_dz))

    def test_min_gives_more_stock_than_avg(self):
        for seed in range(5000, 5030):
            with self.subTest(seed=seed):
                s = Scene(seed)
                r = run_engine(s.path, [PolygonJob(1, "min", s.geometry, BaseLevelOptions.APPROXIMATE_VIA_MIN),
                                        PolygonJob(2, "avg", s.geometry, BaseLevelOptions.APPROXIMATE_VIA_AVG)],
                               s.step_x, s.step_y).results
                if r[1].error or r[2].error:
                    continue
                self.assertLessEqual(r[1].base_z, r[2].base_z + 1e-9)
                self.assertGreaterEqual(r[1].above_m3, r[2].above_m3 - 1e-9)


class NdomConsistencyTest(unittest.TestCase):

    def test_ndom_integral_equals_volume(self):
        for seed in range(6000, 6015):
            with self.subTest(seed=seed):
                s = Scene(seed)
                out = os.path.join(TMP, "ndom_{}.tif".format(seed))
                res = run_engine(s.path, [PolygonJob(1, "p", s.geometry, BaseLevelOptions.MANUAL_BASE_LEVEL, s.level)],
                                 ndom_path=out).results[1]
                data, gt = read_band(out)
                d = data[np.isfinite(data)]
                pixel_area = abs(gt[1] * gt[5])
                # nDOM is float32: compare with a float32-level tolerance
                self.assertAlmostEqual(float(d[d > 0].sum()) * pixel_area, res.above_m3,
                                       delta=1e-5 * max(1.0, res.above_m3))
                self.assertAlmostEqual(float(-d[d < 0].sum()) * pixel_area, res.below_m3,
                                       delta=1e-5 * max(1.0, res.below_m3))
                self.assertEqual(int(d.size), res.cells_valid)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
