# -*- coding: utf-8 -*-
"""
/***************************************************************************
 VolumeCalculationTool - calculation engine
                                 A QGIS plugin
 Calculates volume based on DEM height layers
                              -------------------
        copyright            : (C) 2020-2026 by REDcatch GmbH.
        email                : support@redcatch.at
        modifications        : (C) 2026 by Latitude Drone (calculation engine rewritten)
 ***************************************************************************/

/***************************************************************************
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
 ***************************************************************************/

Pure computation module (no widgets), so it can run inside a QgsTask and be
tested headless.

How the volume is computed
--------------------------
For every polygon a regular grid of cells (step X / step Y) is laid over its
bounding box. The grid is anchored on the DEM pixel grid, so with a step equal
to the pixel size every cell centre is exactly a DEM pixel centre.
A cell belongs to the polygon when its centre lies inside the polygon (GDAL
rasterisation rule). For each such cell:

    dz = height(cell) - base(cell)

and the volume above (resp. below) is the sum of the positive (resp. negative)
dz multiplied by the cell area. Cells where the height DEM or the base DEM is
nodata / outside the raster are skipped and reported as missing coverage.

The base surface is chosen *per polygon*:
    MIN     lowest height sampled along the polygon outline
    AVG     mean height sampled along the polygon outline
    PLANE   least-squares plane fitted to the heights along the outline
    DEM     a second DEM (surface to surface), reprojected on the fly if needed
    MANUAL  a constant level entered by the user
The outline is densified at the DEM pixel size before sampling, so the base is
not limited to the few polygon vertices.
"""

import math
import traceback
from enum import Enum

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsRectangle,
    QgsTask,
    QgsUnitTypes,
    QgsWkbTypes,
)

from .translation import tr, tr_noop

# tr() of a tr_noop() constant (i18n_tool only extracts the literal texts of tr() calls)
_tr_constant = tr

MESSAGE_CATEGORY = "VolumeTask"
NODATA_VALUE = -9999.0
# number of grid cells processed at once (memory ~ 8 bytes * a few arrays)
CHUNK_CELLS = 2_000_000
# upper bound of outline samples per polygon (spacing is enlarged above it)
MAX_CONTOUR_POINTS = 20000
# below this share of valid cells a polygon gets a coverage warning
COVERAGE_WARNING_THRESHOLD = 0.99
# spline base: ground points used for the fit (solve cost grows with the cube)
MAX_SPLINE_POINTS = 1500
# spline base: regularisation in normalised coordinates (0 = interpolate exactly).
# Tuned on synthetic valleys with 0 / 3 / 8 cm DEM noise: 0.1 keeps the curvature
# (error < 0.2 % without noise) while not following the noise (higher values flatten it).
SPLINE_SMOOTHING = 0.1
# stiffness given per polygon, 0 (supple: follows the hollows and bumps of the ground ring) to 10
# (stiff: practically the least-squares plane of the ring); 4 = SPLINE_SMOOTHING
DEFAULT_SPLINE_STIFFNESS = 4.0
SPLINE_STIFFNESS_RANGE = (0.0, 10.0)


def spline_smoothing(stiffness):
    """Regularisation of the thin-plate spline for a stiffness 0..10 (one decade every 2 steps)."""
    lo, hi = SPLINE_STIFFNESS_RANGE
    return 10.0 ** (min(max(float(stiffness), lo), hi) / 2.0 - 3.0)
# spline base: max nodes of the lattice the spline is evaluated on (then bilinear)
SPLINE_LATTICE_NODES = 250
DEFAULT_RING_WIDTH = 3.0


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

class BaseLevelOptions(Enum):
    """Base surface methods. The value is the code stored in the attribute table."""
    APPROXIMATE_VIA_MIN = "MIN"
    APPROXIMATE_VIA_AVG = "AVG"
    BEST_FIT_PLANE = "PLANE"
    TRIANGULATED_OUTLINE = "TIN"
    SPLINE_RING = "SPLINE"
    USE_DEM_LAYER = "DEM"
    MANUAL_BASE_LEVEL = "MANUAL"

    @property
    def code(self):
        return self.value

    @property
    def short_label(self):
        return _tr_constant(_BASE_SHORT_LABELS[self])

    @property
    def explanation(self):
        return _tr_constant(_BASE_EXPLANATIONS[self])

    @classmethod
    def from_code(cls, code):
        if code is None:
            return None
        text = str(code).strip().upper()
        for member in cls:
            if member.value == text:
                return member
        return None

    def __str__(self):
        return self.short_label


# One name per method, used everywhere (table, menus, settings labels, log, messages).
# The long description is in _BASE_EXPLANATIONS (tooltips / info line).
_BASE_SHORT_LABELS = {
    BaseLevelOptions.APPROXIMATE_VIA_MIN: tr_noop("Outline – lowest point"),
    BaseLevelOptions.APPROXIMATE_VIA_AVG: tr_noop("Outline – average"),
    BaseLevelOptions.BEST_FIT_PLANE: tr_noop("Outline – best-fit plane"),
    BaseLevelOptions.TRIANGULATED_OUTLINE: tr_noop("Outline – triangulated (TIN)"),
    BaseLevelOptions.SPLINE_RING: tr_noop("Spline on ground ring"),
    BaseLevelOptions.USE_DEM_LAYER: tr_noop("Base DEM (surface to surface)"),
    BaseLevelOptions.MANUAL_BASE_LEVEL: tr_noop("Manual level"),
}

_BASE_EXPLANATIONS = {
    BaseLevelOptions.APPROXIMATE_VIA_MIN: tr_noop(
        "The base is a horizontal plane at the lowest DEM height found along the polygon outline. "
        "Typical for stockpiles on flat ground; gives the largest stockpile volume."
    ),
    BaseLevelOptions.APPROXIMATE_VIA_AVG: tr_noop(
        "The base is a horizontal plane at the average DEM height along the polygon outline. "
        "Useful when the outline represents a balanced surrounding ground level."
    ),
    BaseLevelOptions.BEST_FIT_PLANE: tr_noop(
        "The base is the inclined plane that best fits (least squares) the DEM heights along the "
        "polygon outline. Recommended for stockpiles or pits on sloped ground."
    ),
    BaseLevelOptions.TRIANGULATED_OUTLINE: tr_noop(
        "The base is a Delaunay triangulation (TIN) of the DEM heights at the polygon vertices (as Pix4D "
        "'Triangulated'): it goes through the points clicked on the natural ground and is straight in between."
    ),
    BaseLevelOptions.SPLINE_RING: tr_noop(
        "The base is a smooth curved surface (thin-plate spline) fitted on the ground in a ring around "
        "the polygon (ring width and stiffness of each polygon set in the 3D view; other checked polygons "
        "are excluded from the ring). It extends the shape of the terrain (valley, bulge, change of slope) "
        "under the stockpile."
    ),
    BaseLevelOptions.USE_DEM_LAYER: tr_noop(
        "The base DEM layer (in 'Settings of the base methods') is used as comparison surface. "
        "Useful for cut/fill, terrain change or comparing two surveys."
    ),
    BaseLevelOptions.MANUAL_BASE_LEVEL: tr_noop(
        "A constant base height entered for the polygon. "
        "Useful when the reference elevation is known."
    ),
}

# Order shown in the combo boxes
BASE_LEVEL_ORDER = [
    BaseLevelOptions.APPROXIMATE_VIA_MIN,
    BaseLevelOptions.APPROXIMATE_VIA_AVG,
    BaseLevelOptions.BEST_FIT_PLANE,
    BaseLevelOptions.TRIANGULATED_OUTLINE,
    BaseLevelOptions.SPLINE_RING,
    BaseLevelOptions.USE_DEM_LAYER,
    BaseLevelOptions.MANUAL_BASE_LEVEL,
]

# Methods that need heights sampled along the polygon outline
OUTLINE_METHODS = (
    BaseLevelOptions.APPROXIMATE_VIA_MIN,
    BaseLevelOptions.APPROXIMATE_VIA_AVG,
    BaseLevelOptions.BEST_FIT_PLANE,
    BaseLevelOptions.TRIANGULATED_OUTLINE,
)


class CountOptions(Enum):
    COUNT_ABOVE_AND_BELOW = 0
    COUNT_ONLY_ABOVE = 1
    COUNT_ONLY_BELOW = 2

    def __str__(self):
        return _tr_constant(_COUNT_LABELS[self])

    @property
    def writes_above(self):
        return self in (CountOptions.COUNT_ABOVE_AND_BELOW, CountOptions.COUNT_ONLY_ABOVE)

    @property
    def writes_below(self):
        return self in (CountOptions.COUNT_ABOVE_AND_BELOW, CountOptions.COUNT_ONLY_BELOW)


_COUNT_LABELS = {
    CountOptions.COUNT_ABOVE_AND_BELOW: tr_noop("Cut and fill (above and below the base)"),
    CountOptions.COUNT_ONLY_ABOVE: tr_noop("Above the base only (stockpile)"),
    CountOptions.COUNT_ONLY_BELOW: tr_noop("Below the base only (pit)"),
}

# key -> (label, size of one unit in m3)
VOLUME_UNITS = {
    "m3": ("m³", 1.0),
    "yd3": ("yd³", 0.764554857984),
    "ft3": ("ft³", 0.028316846592),
}
DEFAULT_VOLUME_UNIT = "m3"


def convert_volume(volume_m3, unit_key):
    return volume_m3 / VOLUME_UNITS[unit_key][1]


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def _meters_unit():
    try:
        return Qgis.DistanceUnit.Meters
    except AttributeError:  # QGIS < 3.30
        return QgsUnitTypes.DistanceMeters


def crs_linear_factor_to_meters(crs):
    """Size of one CRS map unit in metres, or None if not a projected linear unit."""
    if crs is None or not crs.isValid() or crs.isGeographic():
        return None
    try:
        factor = QgsUnitTypes.fromUnitToUnitFactor(crs.mapUnits(), _meters_unit())
    except Exception:
        return None
    if factor is None or not math.isfinite(factor) or factor <= 0:
        return None
    return float(factor)


def crs_unit_abbreviation(crs):
    if crs is None or not crs.isValid():
        return tr("units")
    try:
        text = QgsUnitTypes.toAbbreviatedString(crs.mapUnits())
    except Exception:
        text = ""
    return text or tr("units")


# --------------------------------------------------------------------------
# Raster helpers
# --------------------------------------------------------------------------

_QGIS_DTYPES = {1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32,
                5: np.int32, 6: np.float32, 7: np.float64, 14: np.int8}


def _enum_int(value):
    try:
        return int(value)
    except TypeError:
        return int(value.value)


def block_to_array(block, nrows, ncols):
    """Convert a QgsRasterBlock to a float64 array with NaN for nodata."""
    if block is None or not block.isValid() or block.isEmpty():
        return np.full((nrows, ncols), np.nan)
    if hasattr(block, "as_numpy"):
        arr = block.as_numpy(use_masking=True)
        if np.ma.isMaskedArray(arr):
            data = np.ma.filled(arr.astype(np.float64), np.nan)
        else:
            data = np.array(arr, dtype=np.float64)
        return data.reshape(nrows, ncols)
    # Fallback for older QGIS versions
    dtype = _QGIS_DTYPES.get(_enum_int(block.dataType()))
    if dtype is None:
        raise ValueError("Unsupported raster data type: {}".format(block.dataType()))
    data = np.frombuffer(bytes(block.data()), dtype=dtype).astype(np.float64).reshape(nrows, ncols)
    if block.hasNoDataValue():
        data[data == block.noDataValue()] = np.nan
    elif block.hasNoData():
        for r in range(nrows):
            for c in range(ncols):
                if block.isNoData(r, c):
                    data[r, c] = np.nan
    return data


# max native pixels read at once when sampling a raster
NATIVE_BLOCK_LIMIT = 8_000_000


class RasterGrid:
    """A raster provider with its pixel grid (north-up): origin = top-left corner."""

    def __init__(self, provider, band, origin_x, origin_y, pixel_x, pixel_y):
        self.provider = provider
        self.band = int(band)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.pixel_x = abs(float(pixel_x))
        self.pixel_y = abs(float(pixel_y))

    def indices(self, x, y):
        """Column / row of the pixel containing each point."""
        col = np.floor((np.asarray(x) - self.origin_x) / self.pixel_x).astype(np.int64)
        row = np.floor((self.origin_y - np.asarray(y)) / self.pixel_y).astype(np.int64)
        return col, row

    def sample(self, col, row):
        """Exact nearest-neighbour sampling: value of pixel (row, col) for each index pair
        (2D arrays of the same shape). Pixels are read at native resolution, so no
        resampling of the data provider is involved. Outside the raster / nodata -> NaN."""
        out = np.full(col.shape, np.nan)
        if col.size == 0:
            return out
        self._fill(out, col, row, slice(0, col.shape[0]), slice(0, col.shape[1]))
        return out

    def _fill(self, out, col, row, rs, cs):
        c = col[rs, cs]
        r = row[rs, cs]
        c_lo, c_hi = int(c.min()), int(c.max())
        r_lo, r_hi = int(r.min()), int(r.max())
        width = c_hi - c_lo + 1
        height = r_hi - r_lo + 1
        n_rows = rs.stop - rs.start
        n_cols = cs.stop - cs.start
        if width * height > NATIVE_BLOCK_LIMIT and (n_rows > 1 or n_cols > 1):
            if n_rows >= n_cols:
                mid = rs.start + n_rows // 2
                self._fill(out, col, row, slice(rs.start, mid), cs)
                self._fill(out, col, row, slice(mid, rs.stop), cs)
            else:
                mid = cs.start + n_cols // 2
                self._fill(out, col, row, rs, slice(cs.start, mid))
                self._fill(out, col, row, rs, slice(mid, cs.stop))
            return
        rect = QgsRectangle(self.origin_x + c_lo * self.pixel_x, self.origin_y - (r_hi + 1) * self.pixel_y,
                            self.origin_x + (c_hi + 1) * self.pixel_x, self.origin_y - r_lo * self.pixel_y)
        native = block_to_array(self.provider.block(self.band, rect, width, height), height, width)
        out[rs, cs] = native[r - r_lo, c - c_lo]

    def sample_centres(self, xc, yc):
        """Values at the cell centres of a separable grid (xc: columns, yc: rows)."""
        col, row = self.indices(xc, yc)
        col2d = np.broadcast_to(col[np.newaxis, :], (yc.size, xc.size))
        row2d = np.broadcast_to(row[:, np.newaxis], (yc.size, xc.size))
        return self.sample(col2d, row2d)


class PointTransformer:
    """Vectorised CRS transformation of points (GDAL/PROJ), x/y axis order."""

    def __init__(self, source_wkt, dest_wkt):
        source = osr.SpatialReference()
        source.ImportFromWkt(source_wkt)
        dest = osr.SpatialReference()
        dest.ImportFromWkt(dest_wkt)
        for srs in (source, dest):
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self._transform = osr.CoordinateTransformation(source, dest)

    def __call__(self, x, y):
        shape = np.shape(x)
        points = np.column_stack([np.ravel(x), np.ravel(y)])
        result = np.array(self._transform.TransformPoints(points), dtype=np.float64)
        return result[:, 0].reshape(shape), result[:, 1].reshape(shape)


def grid_window(bbox, origin_x, origin_y, step_x, step_y):
    """Cell window (col0, row0, ncols, nrows) of a grid anchored on (origin_x, origin_y)
    (top-left corner, rows going down) that covers bbox."""
    eps = 1e-9
    c0 = math.floor((bbox.xMinimum() - origin_x) / step_x + eps)
    c1 = math.ceil((bbox.xMaximum() - origin_x) / step_x - eps)
    r0 = math.floor((origin_y - bbox.yMaximum()) / step_y + eps)
    r1 = math.ceil((origin_y - bbox.yMinimum()) / step_y - eps)
    ncols = max(1, c1 - c0)
    nrows = max(1, r1 - r0)
    return c0, r0, ncols, nrows


def estimate_cell_count(geometries, origin_x, origin_y, step_x, step_y):
    total = 0
    for geom in geometries:
        _, _, nc, nr = grid_window(geom.boundingBox(), origin_x, origin_y, step_x, step_y)
        total += nc * nr
    return total


def _ogr_memory_driver():
    for name in ("MEM", "Memory"):
        drv = ogr.GetDriverByName(name)
        if drv is not None and drv.GetMetadataItem("DCAP_VECTOR") == "YES":
            return drv
    drv = ogr.GetDriverByName("Memory")
    if drv is None:
        raise RuntimeError("No OGR in-memory vector driver available")
    return drv


class PolygonMask:
    """Rasterises one polygon on grid windows (cell centre inside polygon rule)."""

    def __init__(self, geometry):
        geom = QgsGeometry(geometry)
        if QgsWkbTypes.isCurvedType(geom.wkbType()):
            geom.convertToStraightSegment()
        self._ds = _ogr_memory_driver().CreateDataSource("mask")
        self._layer = self._ds.CreateLayer("mask", geom_type=ogr.wkbMultiPolygon)
        feature = ogr.Feature(self._layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(bytes(geom.asWkb())))
        self._layer.CreateFeature(feature)
        feature = None

    def rasterize(self, xmin, ymax, step_x, step_y, ncols, nrows):
        ds = gdal.GetDriverByName("MEM").Create("", ncols, nrows, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((xmin, step_x, 0.0, ymax, 0.0, -step_y))
        gdal.RasterizeLayer(ds, [1], self._layer, burn_values=[1])
        mask = ds.GetRasterBand(1).ReadAsArray().astype(bool)
        ds = None
        return mask


def outline_points(geometry, spacing, max_points=MAX_CONTOUR_POINTS):
    """Points along the exterior ring(s) of a (multi)polygon, densified at `spacing`."""
    geom = QgsGeometry(geometry)
    if QgsWkbTypes.isCurvedType(geom.wkbType()):
        geom.convertToStraightSegment()
    if geom.isMultipart():
        polygons = geom.asMultiPolygon()
    else:
        polygons = [geom.asPolygon()]
    rings = [poly[0] for poly in polygons if poly and len(poly[0]) >= 2]
    if not rings:
        return []
    perimeter = sum(QgsGeometry.fromPolylineXY(r).length() for r in rings)
    if spacing <= 0:
        spacing = perimeter / max_points if perimeter > 0 else 1.0
    if perimeter > 0 and perimeter / spacing > max_points:
        spacing = perimeter / max_points
    points = []
    for ring in rings:
        line = QgsGeometry.fromPolylineXY(ring).densifyByDistance(spacing)
        vertices = line.asPolyline()
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]  # closing vertex would be counted twice
        points.extend(vertices)
    return points


def outline_vertices(geometry):
    """Vertices of the exterior ring(s) of a (multi)polygon, as drawn (closing vertex and
    repeated vertices removed)."""
    geom = QgsGeometry(geometry)
    if QgsWkbTypes.isCurvedType(geom.wkbType()):
        geom.convertToStraightSegment()
    polygons = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
    points = []
    seen = set()
    for poly in polygons:
        if not poly:
            continue
        for p in poly[0]:
            key = (p.x(), p.y())
            if key not in seen:
                seen.add(key)
                points.append(p)
    return points


def _gdal_vector_memory_dataset():
    """In-memory vector gdal.Dataset (usable by gdal.Grid)."""
    for name in ("MEM", "Memory"):
        drv = gdal.GetDriverByName(name)
        if drv is not None and drv.GetMetadataItem("DCAP_VECTOR") == "YES":
            ds = drv.Create("", 0, 0, 0, gdal.GDT_Unknown)
            if ds is not None:
                return ds
    raise RuntimeError("No GDAL in-memory vector driver available")


class OutlineTin:
    """Linear interpolation on the Delaunay triangulation of the outline points (gdal.Grid)."""

    NODATA = -1.0e30

    def __init__(self, xs, ys, zs):
        self._ds = _gdal_vector_memory_dataset()
        layer = self._ds.CreateLayer("pts", geom_type=ogr.wkbPoint25D)
        defn = layer.GetLayerDefn()
        for x, y, z in zip(xs, ys, zs):
            feature = ogr.Feature(defn)
            point = ogr.Geometry(ogr.wkbPoint25D)
            point.AddPoint(float(x), float(y), float(z))
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
        feature = None

    def grid(self, xmin, ymax, step_x, step_y, ncols, nrows):
        out = gdal.Grid("", self._ds, format="MEM", outputType=gdal.GDT_Float64,
                        outputBounds=[xmin, ymax, xmin + ncols * step_x, ymax - nrows * step_y],
                        width=ncols, height=nrows, layers=["pts"],
                        algorithm="linear:radius=-1:nodata={}".format(self.NODATA))
        if out is None:
            raise RuntimeError(tr("Triangulation of the outline failed: {}").format(gdal.GetLastErrorMsg()))
        data = out.GetRasterBand(1).ReadAsArray().astype(np.float64)
        out = None
        data[data <= self.NODATA * 0.1] = np.nan
        return data


class ThinPlateSpline:
    """Smoothing thin-plate spline z = f(x, y) (numpy only, coordinates normalised)."""

    def __init__(self, xs, ys, zs, smoothing=SPLINE_SMOOTHING):
        self.cx = float(np.mean(xs))
        self.cy = float(np.mean(ys))
        self.scale = float(max(np.ptp(xs), np.ptp(ys))) or 1.0
        self.u = (np.asarray(xs, dtype=np.float64) - self.cx) / self.scale
        self.v = (np.asarray(ys, dtype=np.float64) - self.cy) / self.scale
        n = self.u.size
        kernel = self._kernel(self.u[:, None] - self.u[None, :], self.v[:, None] - self.v[None, :])
        kernel[np.diag_indices(n)] += smoothing
        poly = np.column_stack([np.ones(n), self.u, self.v])
        system = np.zeros((n + 3, n + 3))
        system[:n, :n] = kernel
        system[:n, n:] = poly
        system[n:, :n] = poly.T
        rhs = np.concatenate([np.asarray(zs, dtype=np.float64), np.zeros(3)])
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        self.weights = solution[:n]
        self.affine = solution[n:]

    @staticmethod
    def _kernel(du, dv):
        r2 = du * du + dv * dv
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 0.5 * r2 * np.log(r2)   # r^2 log r
        k[r2 <= 0] = 0.0
        return k

    def __call__(self, xs, ys):
        u = (np.ravel(xs) - self.cx) / self.scale
        v = (np.ravel(ys) - self.cy) / self.scale
        out = self.affine[0] + self.affine[1] * u + self.affine[2] * v
        block = 4000
        for start in range(0, u.size, block):
            end = start + block
            k = self._kernel(u[start:end, None] - self.u[None, :], v[start:end, None] - self.v[None, :])
            out[start:end] += k @ self.weights
        return out.reshape(np.shape(xs))


class LatticeSurface:
    """A surface evaluated once on a regular lattice, then bilinearly interpolated."""

    def __init__(self, function, bbox, finest_step, max_nodes=SPLINE_LATTICE_NODES):
        size = max(bbox.width(), bbox.height())
        step = max(finest_step, size / float(max_nodes - 1)) if size > 0 else finest_step
        self.x0 = bbox.xMinimum() - step
        self.y0 = bbox.yMinimum() - step
        self.step = step
        nx = int(math.ceil((bbox.xMaximum() + step - self.x0) / step)) + 1
        ny = int(math.ceil((bbox.yMaximum() + step - self.y0) / step)) + 1
        gx = self.x0 + np.arange(nx) * step
        gy = self.y0 + np.arange(ny) * step
        mx, my = np.meshgrid(gx, gy)
        self.values = function(mx, my)  # shape (ny, nx), row 0 = south

    def __call__(self, xc, yc):
        """xc: 1D cell centre x (columns), yc: 1D cell centre y (rows) -> (rows, cols)."""
        ny, nx = self.values.shape
        fx = np.clip((xc - self.x0) / self.step, 0, nx - 1.000001)
        fy = np.clip((yc - self.y0) / self.step, 0, ny - 1.000001)
        ix = np.floor(fx).astype(int)
        iy = np.floor(fy).astype(int)
        tx = (fx - ix)[None, :]
        ty = (fy - iy)[:, None]
        v = self.values
        v00 = v[np.ix_(iy, ix)]
        v01 = v[np.ix_(iy, ix + 1)]
        v10 = v[np.ix_(iy + 1, ix)]
        v11 = v[np.ix_(iy + 1, ix + 1)]
        return (v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty)
                + v10 * (1 - tx) * ty + v11 * tx * ty)


# --------------------------------------------------------------------------
# Jobs, settings and results
# --------------------------------------------------------------------------

class PolygonJob:
    """One polygon to compute. `geometry` must already be in the height DEM CRS."""

    def __init__(self, fid, name, geometry, method, manual_level=0.0, ring_width=None, stiffness=None):
        """ring_width / stiffness: spline base of this polygon (None: CalculationSettings.ring_width /
        DEFAULT_SPLINE_STIFFNESS)."""
        self.fid = fid
        self.name = name
        self.geometry = geometry
        self.method = method
        self.manual_level = float(manual_level)
        self.ring_width = None if ring_width is None else float(ring_width)
        self.stiffness = None if stiffness is None else float(stiffness)


class CalculationSettings:
    def __init__(self, height_provider, height_band, origin_x, origin_y, pixel_x, pixel_y,
                 step_x, step_y, linear_factor, base_provider=None, base_band=1,
                 base_crs_wkts=None, ndom_path=None, ndom_crs_wkt="", ring_width=DEFAULT_RING_WIDTH):
        """
        height_provider / base_provider: cloned raster data providers (thread safe usage)
        origin_x / origin_y: top-left corner of the height DEM (grid anchor)
        linear_factor: size of one CRS unit in metres (heights assumed in the same unit)
        base_crs_wkts: None, or (height DEM CRS WKT, base DEM CRS WKT) when the base DEM
                       is in another CRS: cell centres are then transformed point by point
        """
        self.height_provider = height_provider
        self.height_band = int(height_band)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.pixel_x = abs(float(pixel_x))
        self.pixel_y = abs(float(pixel_y))
        self.step_x = float(step_x)
        self.step_y = float(step_y)
        self.linear_factor = float(linear_factor)
        self.base_provider = base_provider
        self.base_band = int(base_band)
        self.base_crs_wkts = base_crs_wkts
        self.ndom_path = ndom_path
        self.ndom_crs_wkt = ndom_crs_wkt or ""
        self.ring_width = float(ring_width)  # spline base: width of the ground ring (CRS units)


class PolygonResult:
    def __init__(self, fid, name, method):
        self.fid = fid
        self.name = name
        self.method = method
        self.above_m3 = 0.0
        self.below_m3 = 0.0
        self.base_z = None          # constant base level (MIN / AVG / MANUAL)
        self.plane = None           # (a, b, c, xc, yc) for PLANE
        self.cells_inside = 0
        self.cells_valid = 0
        self.outline_samples = 0
        self.fit_points = 0         # points used by PLANE / TIN / SPLINE
        self.fit_rms = None         # RMS residual of the base fit at those points
        self.ring_width = None      # SPLINE: width of the ground ring and stiffness used
        self.stiffness = None
        self.error = None
        self.notes = []

    @property
    def ok(self):
        return self.error is None

    @property
    def coverage(self):
        if self.cells_inside == 0:
            return 0.0
        return self.cells_valid / float(self.cells_inside)

    def base_description(self, unit_abbrev):
        if self.method == BaseLevelOptions.USE_DEM_LAYER:
            return tr("base DEM")
        rms = ""
        if self.fit_rms is not None:
            rms = tr(", fit RMS {:.3f} {}").format(self.fit_rms, unit_abbrev)
        if self.method == BaseLevelOptions.BEST_FIT_PLANE and self.plane is not None:
            a, b, c, _, _ = self.plane
            slope = math.hypot(a, b) * 100.0
            return tr("plane, z={:.3f} {} at outline centre, slope {:.2f} %{}").format(c, unit_abbrev, slope, rms)
        if self.method == BaseLevelOptions.TRIANGULATED_OUTLINE and self.base_z is None:
            return tr("triangulated outline, {} points").format(self.fit_points)
        if self.method == BaseLevelOptions.SPLINE_RING and self.base_z is None:
            return tr("spline on {} ground points, ring {:g} {}, stiffness {:g}{}").format(
                self.fit_points, self.ring_width or 0.0, unit_abbrev, self.stiffness or 0.0, rms)
        if self.base_z is not None:
            return tr("z={:.3f} {}").format(self.base_z, unit_abbrev)
        return "-"


class _Canceled(Exception):
    pass


class _PolygonError(Exception):
    pass


# --------------------------------------------------------------------------
# nDOM writer
# --------------------------------------------------------------------------

class NdomWriter:
    """GeoTIFF on the height DEM pixel grid covering all polygons, filled with dz."""

    def __init__(self, path, settings, col0, row0, ncols, nrows):
        self.col0 = col0
        self.row0 = row0
        self.abs_max = 0.0
        options = ["COMPRESS=LZW", "PREDICTOR=3", "TILED=YES", "BIGTIFF=IF_SAFER"]
        self.ds = gdal.GetDriverByName("GTiff").Create(path, ncols, nrows, 1, gdal.GDT_Float32, options)
        if self.ds is None:
            raise RuntimeError(tr("Cannot create nDOM file: {}").format(path))
        self.ds.SetGeoTransform((settings.origin_x + col0 * settings.pixel_x, settings.pixel_x, 0.0,
                                 settings.origin_y - row0 * settings.pixel_y, 0.0, -settings.pixel_y))
        if settings.ndom_crs_wkt:
            self.ds.SetProjection(settings.ndom_crs_wkt)
        self.band = self.ds.GetRasterBand(1)
        self.band.SetNoDataValue(NODATA_VALUE)
        self.band.Fill(NODATA_VALUE)

    def write(self, dz, valid, col, row):
        x_off = col - self.col0
        y_off = row - self.row0
        nrows, ncols = dz.shape
        existing = self.band.ReadAsArray(x_off, y_off, ncols, nrows)
        out = np.where(valid, dz, existing).astype(np.float32)
        if valid.any():
            self.abs_max = max(self.abs_max, float(np.nanmax(np.abs(dz[valid]))))
        self.band.WriteArray(out, x_off, y_off)

    def close(self):
        if self.ds is not None:
            self.band.FlushCache()
            self.band = None
            self.ds = None


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------

class AccurateVolumeCalculationTask(QgsTask):
    """Computes the volume of each PolygonJob. Results: dict fid -> PolygonResult."""

    def __init__(self, description, jobs, settings, on_finished=None):
        super().__init__(description, QgsTask.CanCancel)
        self.jobs = list(jobs)
        self.settings = settings
        self.on_finished = on_finished
        self.results = {}
        self.exception = None
        self.exception_trace = None
        self.ndom_written = False
        self.ndom_abs_max = 0.0
        self._done_cells = 0
        self._total_cells = 1
        self._height_raster = None  # RasterGrid of the height DEM
        self._base_raster = None    # RasterGrid of the base DEM
        self._base_transform = None  # PointTransformer height CRS -> base CRS
        self._all_polygons = None   # union of the polygons, excluded from the spline rings

    # ---- helpers ---------------------------------------------------------
    def _same_grid_as_dem(self):
        s = self.settings
        return (math.isclose(s.step_x, s.pixel_x, rel_tol=1e-9)
                and math.isclose(s.step_y, s.pixel_y, rel_tol=1e-9))

    def _advance(self, cells):
        self._done_cells += cells
        self.setProgress(min(100.0, 100.0 * self._done_cells / self._total_cells))

    def _prepare_rasters(self):
        """All raster reads go through RasterGrid: exact nearest-neighbour sampling of the
        native pixels (the provider's own resampling is never used)."""
        s = self.settings
        self._height_raster = RasterGrid(s.height_provider, s.height_band,
                                         s.origin_x, s.origin_y, s.pixel_x, s.pixel_y)
        if s.base_provider is None:
            return
        extent = s.base_provider.extent()
        width, height = s.base_provider.xSize(), s.base_provider.ySize()
        if width <= 0 or height <= 0:
            raise RuntimeError(tr("The base DEM has no pixels."))
        self._base_raster = RasterGrid(s.base_provider, s.base_band, extent.xMinimum(), extent.yMaximum(),
                                       extent.width() / width, extent.height() / height)
        if s.base_crs_wkts:
            self._base_transform = PointTransformer(*s.base_crs_wkts)

    def _sample_outline(self, geometry, vertices_only=False):
        """DEM heights along the outline: every pixel along it, or only at the drawn vertices."""
        s = self.settings
        if vertices_only:
            points = outline_vertices(geometry)
        else:
            points = outline_points(geometry, min(s.pixel_x, s.pixel_y))
        if not points:
            return np.array([]), np.array([]), np.array([])
        xs = np.array([p.x() for p in points])
        ys = np.array([p.y() for p in points])
        col, row = self._height_raster.indices(xs, ys)
        zs = self._height_raster.sample(col[np.newaxis, :], row[np.newaxis, :])[0]
        keep = np.isfinite(zs)
        return xs[keep], ys[keep], zs[keep]

    def _base_spec(self, job, res):
        method = job.method
        if method == BaseLevelOptions.MANUAL_BASE_LEVEL:
            res.base_z = job.manual_level
            return ("const", job.manual_level)
        if method == BaseLevelOptions.USE_DEM_LAYER:
            if self._base_raster is None:
                raise _PolygonError(tr("No base DEM layer configured."))
            return ("dem",)
        if method == BaseLevelOptions.SPLINE_RING:
            return self._spline_spec(job, res)

        # TIN: Delaunay triangulation of the drawn vertices only (as Pix4D 'Triangulated'): the user
        # places them on the natural ground; the outline between them may cross a bank / another pile
        vertices_only = method == BaseLevelOptions.TRIANGULATED_OUTLINE
        xs, ys, zs = self._sample_outline(job.geometry, vertices_only)
        res.outline_samples = int(zs.size)
        if zs.size == 0:
            raise _PolygonError(tr("No valid DEM height found along the polygon outline "
                                   "(outline outside the DEM or on nodata)."))
        if method == BaseLevelOptions.APPROXIMATE_VIA_MIN:
            res.base_z = float(zs.min())
            return ("const", res.base_z)
        if method == BaseLevelOptions.APPROXIMATE_VIA_AVG:
            res.base_z = float(zs.mean())
            return ("const", res.base_z)
        if method == BaseLevelOptions.BEST_FIT_PLANE:
            xc, yc = float(xs.mean()), float(ys.mean())
            if zs.size >= 3:
                design = np.column_stack([xs - xc, ys - yc, np.ones_like(xs)])
                coef, _, rank, _ = np.linalg.lstsq(design, zs, rcond=None)
                if rank == 3:
                    a, b, c = (float(v) for v in coef)
                    res.plane = (a, b, c, xc, yc)
                    res.fit_points = int(zs.size)
                    res.fit_rms = float(np.sqrt(np.mean((design @ coef - zs) ** 2)))
                    return ("plane", a, b, c, xc, yc)
            res.base_z = float(zs.mean())
            res.notes.append(tr("plane could not be fitted (degenerate outline), outline average used"))
            return ("const", res.base_z)
        if method == BaseLevelOptions.TRIANGULATED_OUTLINE:
            if zs.size < 3:
                res.base_z = float(zs.mean())
                res.notes.append(tr("too few outline heights to triangulate, outline average used"))
                return ("const", res.base_z)
            res.fit_points = int(zs.size)
            return ("tin", OutlineTin(xs, ys, zs))
        raise _PolygonError(tr("Unknown base level method: {}").format(method))

    def _ring_width(self, job):
        return self.settings.ring_width if job.ring_width is None else job.ring_width

    def _ring_geometry(self, job):
        """Band of ground of the ring width around the polygon, without any included polygon."""
        ring = job.geometry.buffer(self._ring_width(job), 8)
        if self._all_polygons is not None:
            ring = ring.difference(self._all_polygons)
        else:
            ring = ring.difference(job.geometry)
        return ring

    def _spline_spec(self, job, res):
        s = self.settings
        res.ring_width = self._ring_width(job)
        res.stiffness = DEFAULT_SPLINE_STIFFNESS if job.stiffness is None else job.stiffness
        if res.ring_width <= 0:
            raise _PolygonError(tr("The ground ring width for the spline must be greater than 0."))
        ring = self._ring_geometry(job)
        if ring is None or ring.isEmpty():
            raise _PolygonError(tr("No ground left around the polygon for the spline "
                                   "(ring covered by other polygons?)."))
        # sample the ring on a grid fine enough to give ~4x the points used by the fit
        area = max(ring.area(), 1e-9)
        step = max(s.pixel_x, s.pixel_y, math.sqrt(area / (4.0 * MAX_SPLINE_POINTS)))
        c0, r0, ncols, nrows = grid_window(ring.boundingBox(), s.origin_x, s.origin_y, step, step)
        while ncols * nrows > 4 * CHUNK_CELLS:   # very large polygons: coarser sampling
            step *= 2.0
            c0, r0, ncols, nrows = grid_window(ring.boundingBox(), s.origin_x, s.origin_y, step, step)
        xmin = s.origin_x + c0 * step
        ymax = s.origin_y - r0 * step
        mask = PolygonMask(ring).rasterize(xmin, ymax, step, step, ncols, nrows)
        heights = self._height_raster.sample_centres(xmin + (np.arange(ncols) + 0.5) * step,
                                                     ymax - (np.arange(nrows) + 0.5) * step)
        valid = mask & np.isfinite(heights)
        rows, cols = np.nonzero(valid)
        if rows.size < 10:
            raise _PolygonError(tr("Not enough valid DEM heights in the ground ring around the polygon "
                                   "(outside the DEM or nodata) for the spline."))
        if rows.size > MAX_SPLINE_POINTS:
            pick = np.random.default_rng(0).choice(rows.size, MAX_SPLINE_POINTS, replace=False)
            rows, cols = rows[pick], cols[pick]
        xs = xmin + (cols + 0.5) * step
        ys = ymax - (rows + 0.5) * step
        zs = heights[rows, cols]
        spline = ThinPlateSpline(xs, ys, zs, spline_smoothing(res.stiffness))
        res.fit_points = int(zs.size)
        res.fit_rms = float(np.sqrt(np.mean((spline(xs, ys) - zs) ** 2)))
        surface = LatticeSurface(spline, job.geometry.boundingBox(), min(s.step_x, s.step_y))
        return ("lattice", surface)

    def _base_grid(self, spec, rect, xc, yc, nrows, ncols):
        kind = spec[0]
        if kind == "const":
            return np.full((nrows, ncols), spec[1], dtype=np.float64)
        if kind == "plane":
            _, a, b, c, x0, y0 = spec
            return a * (xc[np.newaxis, :] - x0) + b * (yc[:, np.newaxis] - y0) + c
        if kind == "tin":
            step_x = rect.width() / ncols
            step_y = rect.height() / nrows
            return spec[1].grid(rect.xMinimum(), rect.yMaximum(), step_x, step_y, ncols, nrows)
        if kind == "lattice":
            return spec[1](xc, yc)
        # base DEM: pixel containing each cell centre (transformed to the base CRS if needed)
        if self._base_transform is None:
            return self._base_raster.sample_centres(xc, yc)
        gx, gy = np.meshgrid(xc, yc)
        bx, by = self._base_transform(gx, gy)
        col, row = self._base_raster.indices(bx, by)
        return self._base_raster.sample(col, row)

    def _integrate(self, job, spec, res, step_x, step_y, accumulate, ndom):
        s = self.settings
        c0, r0, ncols, nrows = grid_window(job.geometry.boundingBox(), s.origin_x, s.origin_y, step_x, step_y)
        mask_builder = PolygonMask(job.geometry)
        rows_per_chunk = max(1, CHUNK_CELLS // ncols)
        xmin = s.origin_x + c0 * step_x
        xc = s.origin_x + (np.arange(c0, c0 + ncols) + 0.5) * step_x
        above = 0.0
        below = 0.0
        for row in range(r0, r0 + nrows, rows_per_chunk):
            if self.isCanceled():
                raise _Canceled()
            nr = min(rows_per_chunk, r0 + nrows - row)
            ymax = s.origin_y - row * step_y
            mask = mask_builder.rasterize(xmin, ymax, step_x, step_y, ncols, nr)
            if mask.any():
                rect = QgsRectangle(xmin, ymax - nr * step_y, xmin + ncols * step_x, ymax)
                yc = s.origin_y - (np.arange(row, row + nr) + 0.5) * step_y
                height = self._height_raster.sample_centres(xc, yc)
                base = self._base_grid(spec, rect, xc, yc, nr, ncols)
                with np.errstate(invalid="ignore"):
                    dz = height - base
                valid = mask & np.isfinite(dz)
                if accumulate:
                    res.cells_inside += int(mask.sum())
                    res.cells_valid += int(valid.sum())
                    d = dz[valid]
                    above += float(d[d > 0].sum())
                    below += float(-d[d < 0].sum())
                if ndom is not None:
                    ndom.write(dz, valid, c0, row)
            self._advance(ncols * nr)
        if accumulate:
            volume_factor = step_x * step_y * s.linear_factor ** 3
            res.above_m3 = above * volume_factor
            res.below_m3 = below * volume_factor

    def _create_ndom_writer(self):
        s = self.settings
        windows = [grid_window(j.geometry.boundingBox(), s.origin_x, s.origin_y, s.pixel_x, s.pixel_y)
                   for j in self.jobs]
        col0 = min(w[0] for w in windows)
        row0 = min(w[1] for w in windows)
        col1 = max(w[0] + w[2] for w in windows)
        row1 = max(w[1] + w[3] for w in windows)
        return NdomWriter(s.ndom_path, s, col0, row0, col1 - col0, row1 - row0)

    # ---- QgsTask API -----------------------------------------------------
    def run(self):
        QgsMessageLog.logMessage('Started task "{}"'.format(self.description()), MESSAGE_CATEGORY, Qgis.Info)
        s = self.settings
        ndom = None
        try:
            self._prepare_rasters()
            if any(j.method == BaseLevelOptions.SPLINE_RING for j in self.jobs):
                geometries = [j.geometry for j in self.jobs if j.geometry is not None and not j.geometry.isEmpty()]
                if geometries:
                    self._all_polygons = QgsGeometry.unaryUnion(geometries)
            same_grid = self._same_grid_as_dem()
            geometries = [j.geometry for j in self.jobs]
            total = estimate_cell_count(geometries, s.origin_x, s.origin_y, s.step_x, s.step_y)
            if s.ndom_path and self.jobs and not same_grid:
                total += estimate_cell_count(geometries, s.origin_x, s.origin_y, s.pixel_x, s.pixel_y)
            self._total_cells = max(1, total)
            if s.ndom_path and self.jobs:
                ndom = self._create_ndom_writer()

            for job in self.jobs:
                if self.isCanceled():
                    return False
                res = PolygonResult(job.fid, job.name, job.method)
                try:
                    if job.geometry is None or job.geometry.isEmpty():
                        raise _PolygonError(tr("Empty geometry."))
                    spec = self._base_spec(job, res)
                    self._integrate(job, spec, res, s.step_x, s.step_y, True, ndom if same_grid else None)
                    if ndom is not None and not same_grid:
                        self._integrate(job, spec, res, s.pixel_x, s.pixel_y, False, ndom)
                    if res.cells_inside == 0:
                        res.notes.append(tr("polygon smaller than one sampling cell, no cell centre inside"))
                except _PolygonError as exc:
                    res.error = str(exc)
                self.results[job.fid] = res

            if ndom is not None:
                self.ndom_abs_max = ndom.abs_max
                self.ndom_written = True
            return True
        except _Canceled:
            return False
        except Exception as exc:  # reported to the user in finished()
            self.exception = exc
            self.exception_trace = traceback.format_exc()
            QgsMessageLog.logMessage(self.exception_trace, MESSAGE_CATEGORY, Qgis.Critical)
            return False
        finally:
            if ndom is not None:
                ndom.close()

    def finished(self, result):
        if result:
            QgsMessageLog.logMessage('Task "{}" completed'.format(self.description()), MESSAGE_CATEGORY, Qgis.Success)
        if self.on_finished is not None:
            try:
                self.on_finished(self, result)
            except Exception:  # never let an exception escape a C++ virtual
                QgsMessageLog.logMessage(traceback.format_exc(), MESSAGE_CATEGORY, Qgis.Critical)

    def cancel(self):
        QgsMessageLog.logMessage('Task "{}" was canceled'.format(self.description()), MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()
