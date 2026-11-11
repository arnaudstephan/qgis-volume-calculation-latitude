# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 Latitude Drone - GNU GPL v2 or later (see LICENSE).
3D view of the base surface under one polygon.

Shows the DEM over the polygon and a band of ground around it, with the base surface
of a chosen method drawn under the stock, so the user can judge whether the shape of
the base fits the ground of this stock. The base and the volume come from the
calculation engine itself (same code as the real calculation, nothing re-implemented).
"""

import html
import math
import os
import traceback

import numpy as np
from qgis.PyQt.QtCore import QLocale, QSize, Qt, QTimer
from qgis.PyQt.QtGui import QIcon, QKeySequence
from qgis.PyQt.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                                 QSizePolicy, QSlider, QToolButton, QVBoxLayout)
try:
    from qgis.PyQt.QtWidgets import QShortcut   # Qt 5
except ImportError:  # pragma: no cover
    from qgis.PyQt.QtGui import QShortcut       # Qt 6
from qgis.core import Qgis, QgsApplication, QgsGeometry, QgsMessageLog, QgsRectangle

from . import theme
from .translation import tr
from .volume_engine import (BASE_LEVEL_ORDER, DEFAULT_SPLINE_STIFFNESS, MESSAGE_CATEGORY, SPLINE_STIFFNESS_RANGE,
                            VOLUME_UNITS, AccurateVolumeCalculationTask, BaseLevelOptions, PolygonJob,
                            PolygonMask, PolygonResult, convert_volume, outline_points)

# nodes per side of the display grid (the volume itself is computed at the real sampling step)
MAX_DISPLAY_NODES = 90
# nodes per side of the lighter mesh shown while the 3D view is being rotated
ROTATION_DISPLAY_NODES = 30
# perspective of the 3D view: focal length of matplotlib (1 = 90 degrees field of view, smaller = stronger)
PERSPECTIVE_FOCAL_LENGTH = 0.3
# largest ring width reachable with the slider (the field accepts more)
RING_SLIDER_MAX = 30.0
# delay before the base is recomputed while a manual level is being typed (ms)
LEVEL_DEBOUNCE_MS = 350

STOCK_TRANSPARENT, STOCK_OPAQUE, STOCK_HIDDEN = "transparent", "opaque", "hidden"
BASE_BELOW_GROUND = (0.13, 0.42, 0.85, 0.65)   # blue: base under the DEM (counted above)
BASE_ABOVE_GROUND = (0.85, 0.13, 0.13, 0.75)   # red: base above the DEM (counted below)
GROUND_COLOR = np.array([0.72, 0.72, 0.70])
LIGHT = np.array([-1.0, 1.0, 2.0]) / math.sqrt(6.0)       # from the north-west, 55 degrees high
# glossy highlight of the base: half vector between the light and a viewer above the scene
HALF_VECTOR = (LIGHT + np.array([0.0, 0.0, 1.0])) / np.linalg.norm(LIGHT + np.array([0.0, 0.0, 1.0]))
BASE_GLOSS = 0.55
BASE_SHININESS = 30.0
PROFILE_COLORS = {"A": "#c51b7d", "B": "#1b7837"}
PROFILE_COLORS_DARK = {"A": "#f06bb4", "B": "#5cc46a"}    # same hues, readable on a dark theme

# "cards" around the two figures, colour derived from the palette (light and dark QGIS themes)
PREVIEW_STYLE = """
QFrame#card { background: %(card)s; border: 1px solid palette(midlight); border-radius: 8px; }
QLabel#polygonTitle { font-weight: 600; }
"""


def _hex(rgb):
    """'#rrggbb' of an (r, g, b[, a]) colour in 0..1."""
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * float(v))) for v in list(rgb)[:3]))


def _separator(parent):
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _card(widget, parent):
    """`widget` in a rounded frame with the base colour of the palette."""
    frame = QFrame(parent)
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.addWidget(widget)
    return frame


def _matplotlib_font():
    """Font family of the Qt interface if matplotlib has it (same look as the dialog), else None."""
    try:
        from matplotlib import font_manager
        family = QApplication.font().family()
        font_manager.findfont(family, fallback_to_default=False)
        return family
    except Exception:
        return None


class PreviewError(Exception):
    """The base of this method cannot be built for the polygon (message for the user)."""


class BasePreviewComputer:
    """Display grids (DEM, base of each method) and exact volumes of one polygon."""

    def __init__(self, fid, name, geometry, settings, all_polygons, manual_level_getter, base_problem=None,
                 spline_params=None):
        """spline_params: (ring width, stiffness) of the polygon's spline base."""
        self.fid = fid
        self.base_problem = base_problem   # why the 'Base DEM' method cannot be shown (no usable layer)
        self.name = name
        self.geometry = geometry
        self.settings = settings
        self._manual_level = manual_level_getter
        self.spline_params = tuple(spline_params or (settings.ring_width, DEFAULT_SPLINE_STIFFNESS))
        self._task = AccurateVolumeCalculationTask("Base preview", [], settings)
        self._task._prepare_rasters()
        self._task._all_polygons = all_polygons   # excluded from the spline ground ring
        self._cache = {}
        self._ring_outlines = {}
        self._build_display_grid(self.spline_params[0])

    def _sampled_outlines(self, geometry):
        """Exterior ring(s) of `geometry` as (x, y, DEM height) arrays, for drawing."""
        raster = self._task._height_raster
        out = []
        parts = geometry.asGeometryCollection() if geometry.isMultipart() else [geometry]
        for part in parts:
            points = outline_points(QgsGeometry(part), self.step / 2.0)
            if len(points) < 2:
                continue
            px = np.array([p.x() for p in points] + [points[0].x()])
            py = np.array([p.y() for p in points] + [points[0].y()])
            col, row = raster.indices(px, py)
            pz = raster.sample(col[np.newaxis, :], row[np.newaxis, :])[0]
            out.append((px, py, pz))
        return out

    def ringOutlines(self, ring_width):
        """Outer edge of the spline ground ring, for drawing."""
        key = round(float(ring_width), 6)
        if key not in self._ring_outlines:
            self._ring_outlines[key] = self._sampled_outlines(self.geometry.buffer(float(ring_width), 8))
        return self._ring_outlines[key]

    def _build_display_grid(self, ring_width):
        s = self.settings
        bbox = self.geometry.boundingBox()
        size = max(bbox.width(), bbox.height(), s.pixel_x)
        # enough ground around the stock to judge the base, and the whole spline ring
        margin = max(0.25 * size, float(ring_width) * 1.25)
        self.margin = margin
        xmin, xmax = bbox.xMinimum() - margin, bbox.xMaximum() + margin
        ymin, ymax = bbox.yMinimum() - margin, bbox.yMaximum() + margin
        step = max(s.pixel_x, s.pixel_y, max(xmax - xmin, ymax - ymin) / (MAX_DISPLAY_NODES - 1))
        nx = int(math.floor((xmax - xmin) / step)) + 1
        ny = int(math.floor((ymax - ymin) / step)) + 1
        self.step = step
        self.xs = xmin + np.arange(nx) * step
        self.ys = ymax - np.arange(ny) * step        # rows go south
        self.cx = bbox.center().x()
        self.cy = bbox.center().y()
        raster = self._task._height_raster
        self.terrain = raster.sample_centres(self.xs, self.ys)
        # nodes inside the polygon (a node is the centre of a cell of a grid shifted by half a step)
        self.inside = PolygonMask(self.geometry).rasterize(xmin - step / 2, ymax + step / 2, step, step, nx, ny)
        self.outlines = self._sampled_outlines(self.geometry)
        self._cache = {}          # base grids are on the display grid
        self._ring_outlines = {}

    def initialManualLevel(self):
        """Level of the row if one was entered, otherwise the lowest point of the outline."""
        level = self._manual_level()
        if level is not None:
            return level
        heights = np.concatenate([pz for _, _, pz in self.outlines]) if self.outlines else np.array([])
        heights = heights[np.isfinite(heights)]
        if heights.size:
            return round(float(heights.min()), 2)
        finite = self.terrain[np.isfinite(self.terrain)]
        return round(float(finite.min()), 2) if finite.size else 0.0

    def compute(self, method, level=None, spline=None):
        """dict(base=grid of the base at the display nodes, result=PolygonResult with the exact volumes).
        level: the constant base of 'Manual level'; spline: (ring width, stiffness) of 'Spline'."""
        key = method
        ring = stiffness = None
        if method == BaseLevelOptions.USE_DEM_LAYER and self.settings.base_provider is None:
            raise PreviewError(self.base_problem or tr("No usable base DEM layer."))
        if method == BaseLevelOptions.MANUAL_BASE_LEVEL:
            if level is None:
                raise PreviewError(tr("Type the level of the base."))
            level = round(float(level), 6)
            key = (method, level)
        else:
            level = None
        if method == BaseLevelOptions.SPLINE_RING:
            ring, stiffness = (round(float(v), 6) for v in (spline or self.spline_params))
            key = (method, ring, stiffness)
            if ring * 1.25 > self.margin + 1e-9:   # wider ring: show all of it
                self._build_display_grid(ring)
        if key in self._cache:
            return self._cache[key]
        job = PolygonJob(self.fid, self.name, self.geometry, method, level or 0.0,
                         ring_width=ring, stiffness=stiffness)
        res = PolygonResult(self.fid, self.name, method)
        task = self._task
        try:
            spec = task._base_spec(job, res)
        except Exception as exc:   # _PolygonError: explained to the user
            raise PreviewError(str(exc))
        ny, nx = self.terrain.shape
        half = self.step / 2.0
        rect = QgsRectangle(self.xs[0] - half, self.ys[-1] - half, self.xs[-1] + half, self.ys[0] + half)
        base = np.asarray(task._base_grid(spec, rect, self.xs, self.ys, ny, nx), dtype=np.float64)
        s = self.settings
        task._integrate(job, spec, res, s.step_x, s.step_y, True, None)
        data = {"base": base, "result": res}
        self._cache[key] = data
        return data


def _fast_mesh(faces, colors):
    """Poly3DCollection whose depth sorting is vectorised. Matplotlib's own one handles the faces
    one by one with masked arrays (about 0.2 ms per face: seconds per frame while rotating)."""
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    class FastQuadMesh(Poly3DCollection):
        def __init__(self, faces, colors):
            super().__init__(faces, facecolors=colors, edgecolors="none", linewidths=0)
            self._faces = np.asarray(faces, dtype=np.float64)       # (n, 4, 3)
            self._colors = np.asarray(colors, dtype=np.float64)     # (n, 4) RGBA

        def do_3d_projection(self):
            n = self._faces.shape[0]
            if n == 0:
                return np.nan
            points = self._faces.reshape(-1, 3)
            xs, ys, zs = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], self.axes.M)
            depth = np.asarray(zs).reshape(n, 4).mean(axis=1)
            order = np.argsort(depth)[::-1]       # farthest first (painter's algorithm)
            verts = np.stack([np.asarray(xs), np.asarray(ys)], axis=-1).reshape(n, 4, 2)[order]
            PolyCollection.set_verts(self, verts, closed=False)
            # colours actually drawn by Poly3DCollection (get_facecolor / get_edgecolor)
            self._facecolors2d = self._colors[order]
            self._edgecolors2d = np.zeros((0, 4))
            return float(np.min(depth))

    return FastQuadMesh(faces, colors)


def _quad_corners(grid):
    """(ny-1, nx-1, 4) values at the 4 corners of every quad of a node grid."""
    return np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=-1)


def _normals(px, py, pz, exaggeration):
    """Unit upward normal of each quad (corner arrays (n, 4)), with the displayed vertical scale."""
    pz = pz * exaggeration
    d1 = np.stack([px[:, 2] - px[:, 0], py[:, 2] - py[:, 0], pz[:, 2] - pz[:, 0]], axis=1)
    d2 = np.stack([px[:, 3] - px[:, 1], py[:, 3] - py[:, 1], pz[:, 3] - pz[:, 1]], axis=1)
    normal = np.cross(d1, d2)
    length = np.linalg.norm(normal, axis=1)
    length[length == 0] = 1.0
    normal /= length[:, None]
    normal[normal[:, 2] < 0] *= -1.0
    return normal


def _shade(px, py, pz, exaggeration):
    """Lambert shading factor of each quad (ground and stock)."""
    return 0.45 + 0.55 * np.clip(_normals(px, py, pz, exaggeration) @ LIGHT, 0.0, 1.0)


def _lit(rgb, px, py, pz, exaggeration):
    """Base surface: contrasted diffuse light + glossy highlight (Blinn-Phong), so its curvature
    reads at a glance. Computed once per drawing (colours of the faces): no cost while rotating."""
    normal = _normals(px, py, pz, exaggeration)
    diffuse = 0.30 + 0.70 * np.clip(normal @ LIGHT, 0.0, 1.0)
    specular = BASE_GLOSS * np.clip(normal @ HALF_VECTOR, 0.0, 1.0) ** BASE_SHININESS
    return np.clip(rgb * diffuse[:, None] + specular[:, None], 0.0, 1.0)


class BasePreviewDialog(QDialog):
    """Non-modal window: the stock in 3D with the base of the chosen method."""

    def __init__(self, computer, method, unit_key, decimals, dem_unit, title, apply_method, parent=None,
                 navigate=None):
        """apply_method(fid, method, level) -> bool: sets the method in the row of the table.
        navigate(fid, step) -> (computer, method, title) of the previous (-1) / next (+1) polygon, or None."""
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.computer = computer
        self.unit_key = unit_key
        self.decimals = decimals
        self.dem_unit = dem_unit
        self.apply_method = apply_method
        self.navigate = navigate
        self._drawn = False
        self._data = None
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")))
        self.resize(1250, 800)

        self.comboMethod = QComboBox(self)
        for option in BASE_LEVEL_ORDER:
            self.comboMethod.addItem(option.short_label, option.code)
            self.comboMethod.setItemData(self.comboMethod.count() - 1, option.explanation, Qt.ItemDataRole.ToolTipRole)
        self.comboMethod.setCurrentIndex(max(0, self.comboMethod.findData(method.code)))
        self.comboStock = QComboBox(self)
        self.comboStock.addItem(tr("Transparent"), STOCK_TRANSPARENT)
        self.comboStock.addItem(tr("Opaque"), STOCK_OPAQUE)
        self.comboStock.addItem(tr("Hidden"), STOCK_HIDDEN)
        self.comboStock.setToolTip(tr("How the surface of the stock (DEM inside the polygon) is drawn"))
        self.spinExaggeration = QDoubleSpinBox(self)
        self.spinExaggeration.setRange(1.0, 10.0)
        self.spinExaggeration.setSingleStep(0.5)
        self.spinExaggeration.setDecimals(1)
        self.spinExaggeration.setValue(1.0)   # true scale by default
        self.spinExaggeration.setSuffix(" ×")
        self.spinExaggeration.setToolTip(tr("Vertical exaggeration: makes slopes and bumps of the base easier to see"))
        self.labelLevel = QLabel(tr("Level"), self)
        self.spinLevel = QDoubleSpinBox(self)
        self.spinLevel.setRange(-10000.0, 100000.0)
        self.spinLevel.setDecimals(3)
        self.spinLevel.setSingleStep(0.1)
        self.spinLevel.setSuffix(" " + dem_unit)
        self.spinLevel.setValue(computer.initialManualLevel())
        self.spinLevel.setToolTip(tr("Height of the horizontal base"))
        # spline base of this polygon: width of the ground ring and stiffness
        ring, stiffness = computer.spline_params
        self.labelRing = QLabel(tr("Ring"), self)
        self.spinRing = QDoubleSpinBox(self)
        self.spinRing.setRange(0.5, 200.0)
        self.spinRing.setDecimals(1)
        self.spinRing.setSingleStep(0.5)
        self.spinRing.setSuffix(" " + dem_unit)
        self.spinRing.setValue(ring)
        self.spinRing.setToolTip(tr("Width of the band of ground around the polygon on which the spline is fitted "
                                    "(dotted line)"))
        self.labelStiffness = QLabel(tr("Stiffness"), self)
        self.spinStiffness = QDoubleSpinBox(self)
        self.spinStiffness.setRange(*SPLINE_STIFFNESS_RANGE)
        self.spinStiffness.setDecimals(1)
        self.spinStiffness.setSingleStep(1.0)
        self.spinStiffness.setValue(stiffness)
        self.spinStiffness.setToolTip(tr("0: supple, follows the hollows and bumps of the ring — 10: stiff, "
                                         "nearly an inclined plane"))
        # sliders next to the fields (half steps); the field keeps the exact value
        self.sliderRing = self._sliderFor(self.spinRing, 2, (0.5, RING_SLIDER_MAX))
        self.sliderStiffness = self._sliderFor(self.spinStiffness, 2, SPLINE_STIFFNESS_RANGE)
        self._levelTimer = QTimer(self)
        self._levelTimer.setSingleShot(True)
        self._levelTimer.setInterval(LEVEL_DEBOUNCE_MS)

        # previous / next polygon of the table, without going back to the main window
        # (QGIS arrow icons: "◀ ▶" characters are missing in some fonts)
        self.buttonPrevious = QToolButton(self)
        self.buttonNext = QToolButton(self)
        self.buttonPrevious.setIcon(QgsApplication.getThemeIcon("/mActionArrowLeft.svg"))
        self.buttonNext.setIcon(QgsApplication.getThemeIcon("/mActionArrowRight.svg"))
        self.buttonPrevious.setToolTip(tr("Previous polygon of the table (Alt+Left)"))
        self.buttonNext.setToolTip(tr("Next polygon of the table (Alt+Right)"))
        for button, key in ((self.buttonPrevious, "Alt+Left"), (self.buttonNext, "Alt+Right")):
            button.setAutoRaise(True)
            button.setEnabled(navigate is not None)
            QShortcut(QKeySequence(key), self, activated=button.click)
        self.labelPolygon = QLabel(self)
        self.labelPolygon.setObjectName("polygonTitle")
        self.labelPolygon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelPolygon.setMinimumWidth(self.labelPolygon.fontMetrics().horizontalAdvance("ID 00000") + 16)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.buttonPrevious)
        controls.addWidget(self.labelPolygon)
        controls.addWidget(self.buttonNext)
        controls.addWidget(_separator(self))
        labelMethod = QLabel(tr("Base"), self)
        labelMethod.setToolTip(tr("Base method of this polygon: chosen here, it is kept for the polygon"))
        controls.addWidget(labelMethod)
        controls.addWidget(self.comboMethod)
        controls.addSpacing(6)
        controls.addWidget(self.labelLevel)
        controls.addWidget(self.spinLevel)
        controls.addWidget(self.labelRing)
        controls.addWidget(self.sliderRing)
        controls.addWidget(self.spinRing)
        controls.addSpacing(6)
        controls.addWidget(self.labelStiffness)
        controls.addWidget(self.sliderStiffness)
        controls.addWidget(self.spinStiffness)
        controls.addStretch(1)
        controls.addWidget(_separator(self))
        controls.addWidget(QLabel(tr("Stock"), self))
        controls.addWidget(self.comboStock)
        labelExaggeration = QLabel("Z", self)
        labelExaggeration.setToolTip(self.spinExaggeration.toolTip())
        controls.addSpacing(6)
        controls.addWidget(labelExaggeration)
        controls.addWidget(self.spinExaggeration)

        # volumes of the base shown (large), and what happened to the choice (saved...)
        self.labelInfo = QLabel(self)
        self.labelInfo.setWordWrap(True)
        self.labelInfo.setTextFormat(Qt.TextFormat.RichText)
        self.labelInfo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.labelNote = QLabel(self)
        self.labelNote.setTextFormat(Qt.TextFormat.RichText)
        self.labelNote.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        summary = QHBoxLayout()
        summary.setContentsMargins(4, 0, 4, 0)
        summary.addWidget(self.labelInfo, 1)
        summary.addWidget(self.labelNote)

        # two canvases: rotating the 3D view only redraws the 3D one, not the cross-sections
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setToolTip(tr("Drag to rotate, right-drag to zoom"))
        self.figureProfiles = Figure(figsize=(4, 6))
        self.canvasProfiles = FigureCanvasQTAgg(self.figureProfiles)
        self.canvasProfiles.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = None
        self._fineMesh = None     # Poly3DCollection shown at rest
        self._coarseMesh = None   # lighter one, shown while rotating
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setIconSize(QSize(18, 18))

        palette = theme.colors(self)
        legend = QLabel(
            "<span style='color:{};'>■</span>&nbsp;{} &nbsp;&nbsp; <span style='color:{};'>■</span>&nbsp;{} &nbsp;&nbsp; "
            "<span style='color:{};'>■</span>&nbsp;{} &nbsp;&nbsp; <b>—</b>&nbsp;{}".format(
                _hex(BASE_BELOW_GROUND), html.escape(tr("base below the ground")),
                _hex(BASE_ABOVE_GROUND), html.escape(tr("base above the ground")),
                _hex(GROUND_COLOR), html.escape(tr("ground")), html.escape(tr("outline"))), self)
        legend.setStyleSheet("color: {};".format(palette["muted"]))
        legend.setToolTip(tr("Blue: stock volume (above the base). Red: counted below the base. "
                             "Dashed lines: the two cross-sections."))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addLayout(summary)
        views = QHBoxLayout()
        views.setSpacing(10)
        views.addWidget(_card(self.canvas, self), 3)
        views.addWidget(_card(self.canvasProfiles, self), 2)
        layout.addLayout(views, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(toolbar)
        bottom.addStretch(1)
        bottom.addWidget(legend)
        layout.addLayout(bottom)
        self.setStyleSheet(PREVIEW_STYLE % {"card": palette["card"]})

        # a method / level chosen here is the method kept for the polygon (table + layer)
        self.comboMethod.currentIndexChanged.connect(lambda _i: self.refresh(adopt=True))
        for spin in (self.spinLevel, self.spinRing, self.spinStiffness):
            spin.valueChanged.connect(lambda _v: self._levelTimer.start())
        self._levelTimer.timeout.connect(lambda: self.refresh(adopt=True))
        self.canvas.mpl_connect("button_press_event", self._onMousePress)
        self.canvas.mpl_connect("button_release_event", self._onMouseRelease)
        self.comboStock.currentIndexChanged.connect(lambda _i: self.draw())
        self.spinExaggeration.valueChanged.connect(lambda _v: self.draw())
        self.buttonPrevious.clicked.connect(lambda: self._navigate(-1))
        self.buttonNext.clicked.connect(lambda: self._navigate(1))
        self.labelPolygon.setText(tr("ID {}").format(computer.fid))
        self.labelPolygon.setToolTip(computer.name or "")
        self.refresh()

    # ------------------------------------------------------------------ data
    def currentMethod(self):
        return BaseLevelOptions.from_code(self.comboMethod.currentData())

    def _flushLevel(self):
        """A level typed a moment ago (not taken yet) is taken before leaving the polygon."""
        if self._levelTimer.isActive():
            self._levelTimer.stop()
            self.refresh(adopt=True)

    def closeEvent(self, event):
        self._flushLevel()
        super().closeEvent(event)

    def _navigate(self, step):
        if self.navigate is None:
            return
        self._flushLevel()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = self.navigate(self.computer.fid, step)
        finally:
            QApplication.restoreOverrideCursor()
        if found is not None:
            self.showPolygon(*found)

    def showPolygon(self, computer, method, title):
        """Show another polygon in this window (view angle, stock mode and exaggeration are kept)."""
        self._levelTimer.stop()
        self.computer = computer
        self._data = None
        self.setWindowTitle(title)
        self.labelPolygon.setText(tr("ID {}").format(computer.fid))
        self.labelPolygon.setToolTip(computer.name or "")
        widgets = (self.comboMethod, self.spinLevel, self.spinRing, self.spinStiffness)
        for widget in widgets:
            widget.blockSignals(True)
        self.comboMethod.setCurrentIndex(max(0, self.comboMethod.findData(method.code)))
        self.spinLevel.setValue(computer.initialManualLevel())
        self.spinRing.setValue(computer.spline_params[0])
        self.spinStiffness.setValue(computer.spline_params[1])
        for widget in widgets:
            widget.blockSignals(False)
        self._syncSliders()
        self.refresh()

    def splineParams(self):
        return self.spinRing.value(), self.spinStiffness.value()

    def _sliderFor(self, spin, factor, value_range):
        """Horizontal slider bound to `spin` (slider units = value * factor). Dragging updates the
        field (computed after a short pause, as when typing); releasing computes at once."""
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(int(round(value_range[0] * factor)), int(round(value_range[1] * factor)))
        slider.setFixedWidth(110)
        slider.setToolTip(spin.toolTip())
        slider.setProperty("factor", factor)
        slider.setValue(int(round(spin.value() * factor)))
        slider.valueChanged.connect(lambda v, s=spin: s.setValue(v / float(factor)))
        slider.sliderReleased.connect(self._flushLevel)

        def follow(value, sl=slider):
            sl.blockSignals(True)
            sl.setValue(int(round(value * factor)))
            sl.blockSignals(False)
        spin.valueChanged.connect(follow)
        self._sliders = getattr(self, "_sliders", []) + [(slider, spin)]
        return slider

    def _syncSliders(self):
        for slider, spin in self._sliders:
            slider.blockSignals(True)
            slider.setValue(int(round(spin.value() * slider.property("factor"))))
            slider.blockSignals(False)

    def refresh(self, adopt=False):
        """Compute and draw the base of the chosen method. adopt: the user chose it here, it becomes
        the method of the polygon (only when it can be built for this polygon)."""
        method = self.currentMethod()
        is_manual = method == BaseLevelOptions.MANUAL_BASE_LEVEL
        is_spline = method == BaseLevelOptions.SPLINE_RING
        self.labelLevel.setVisible(is_manual)
        self.spinLevel.setVisible(is_manual)
        for widget in (self.labelRing, self.sliderRing, self.spinRing,
                       self.labelStiffness, self.sliderStiffness, self.spinStiffness):
            widget.setVisible(is_spline)
        error = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._data = self.computer.compute(method, self.spinLevel.value() if is_manual else None,
                                               self.splineParams() if is_spline else None)
        except PreviewError as exc:
            self._data, error = None, str(exc)
        except Exception as exc:
            QgsMessageLog.logMessage(traceback.format_exc(), MESSAGE_CATEGORY, Qgis.Critical)
            self._data, error = None, tr("Unexpected error: {}").format(exc)
        finally:
            QApplication.restoreOverrideCursor()
        self._updateInfo(method, error)
        if adopt and self._data is not None:
            self._adopt(method)
        self.draw()

    def _fmt(self, volume_m3):
        """Volume in the output unit, written as in the table (decimal separator of the locale)."""
        return "{} {}".format(QLocale().toString(convert_volume(volume_m3, self.unit_key), "f", self.decimals),
                              VOLUME_UNITS[self.unit_key][0])

    def _updateInfo(self, method, error, note=""):
        """Volumes above / below the base in large, the base in small (the explanation of the
        method is the tooltip of its combo box); `note`: what happened to the choice, on the right."""
        colors = theme.colors(self)
        parts = []
        if error:
            parts.append("<span style='color:{};'><b>{}</b> {}</span>".format(
                theme.ERROR_COLOR, html.escape(tr("Not possible for this polygon:")), html.escape(error)))
        elif self._data is not None:
            res = self._data["result"]
            figures = []
            for volume, caption in ((res.above_m3, tr("above the base")), (res.below_m3, tr("below the base"))):
                figures.append("<span style='font-size:15pt; font-weight:600;'>{}</span>"
                               "<span style='color:{};'>&nbsp; {}</span>".format(
                                   html.escape(self._fmt(volume)), colors["muted"], html.escape(caption)))
            figures.append("<span style='color:{};'>{}</span>".format(
                colors["muted"], html.escape(tr("base: {}").format(res.base_description(self.dem_unit)))))
            parts.append(" &nbsp;&nbsp;&nbsp;&nbsp; ".join(figures))
            if res.cells_inside and res.coverage < 0.99:
                parts.append("<span style='color:{};'>{}</span>".format(theme.ERROR_COLOR, html.escape(
                    tr("Only {:.1f} % of the polygon has DEM data.").format(100.0 * res.coverage))))
            for text in res.notes:
                parts.append("<i>{}</i>".format(html.escape(text)))
        self.labelInfo.setText("<br>".join(parts))
        if note:
            self.labelNote.setText("<span style='color:{};'>{}</span>".format(colors["muted"], html.escape(note)))
        else:
            self.labelNote.clear()

    def _adopt(self, method):
        """The method (and level) chosen here becomes the one of the polygon: row of the table + layer."""
        level = self.spinLevel.value() if method == BaseLevelOptions.MANUAL_BASE_LEVEL else None
        spline = self.splineParams() if method == BaseLevelOptions.SPLINE_RING else None
        note = self.apply_method(self.computer.fid, method, level, spline)
        if note:
            self._updateInfo(method, None, note)

    # ------------------------------------------------------------------ rotation
    def _onMousePress(self, event):
        """Lighter mesh while the view is dragged: the rotation stays fluid."""
        if event.inaxes is self.ax and self._coarseMesh is not None:
            self._fineMesh.set_visible(False)
            self._coarseMesh.set_visible(True)

    def _onMouseRelease(self, _event):
        if self._coarseMesh is not None and self._coarseMesh.get_visible():
            self._coarseMesh.set_visible(False)
            self._fineMesh.set_visible(True)
            self.canvas.draw()

    # ------------------------------------------------------------------ drawing
    def _style(self):
        """Matplotlib settings matching the dialog: font of the interface, colours of the palette."""
        colors = theme.colors(self)
        rc = {"text.color": colors["text"], "axes.labelcolor": colors["muted"], "axes.edgecolor": colors["border"],
              "xtick.color": colors["muted"], "ytick.color": colors["muted"], "xtick.labelsize": 7.5,
              "ytick.labelsize": 7.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "grid.color": colors["grid"],
              "grid.linewidth": 0.6, "axes.facecolor": colors["card"], "figure.facecolor": colors["card"],
              "savefig.facecolor": colors["card"]}
        if not hasattr(self, "_font"):
            self._font = _matplotlib_font()
        if self._font:
            rc["font.family"] = [self._font, "DejaVu Sans"]
        return colors, rc

    def draw(self):
        import matplotlib
        colors, rc = self._style()
        with matplotlib.rc_context(rc):
            self._draw(colors)

    def _draw(self, colors):
        c = self.computer
        profile_colors = PROFILE_COLORS_DARK if colors["dark"] else PROFILE_COLORS
        exaggeration = self.spinExaggeration.value()
        stock_mode = self.comboStock.currentData()
        if self._drawn and self.ax is not None:
            elev, azim = self.ax.elev, self.ax.azim
        else:
            elev, azim = 30.0, -60.0
        self.figure.clear()
        self.figure.set_facecolor(colors["card"])
        self.figureProfiles.set_facecolor(colors["card"])
        self.figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.03)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_proj_type("persp", focal_length=PERSPECTIVE_FOCAL_LENGTH)
        self._style3d(colors)
        base = self._data["base"] if self._data is not None else None
        row = int(np.argmin(np.abs(c.ys - c.cy)))
        col = int(np.argmin(np.abs(c.xs - c.cx)))
        self.figureProfiles.clear()
        grid = self.figureProfiles.add_gridspec(2, 1, left=0.12, right=0.97, top=0.92, bottom=0.08, hspace=0.42)
        self._drawProfile(self.figureProfiles.add_subplot(grid[0]), "A", tr("west → east"), c.xs - c.cx,
                          c.terrain[row, :], None if base is None else base[row, :], c.inside[row, :], colors)
        order = np.argsort(c.ys)   # south → north
        self._drawProfile(self.figureProfiles.add_subplot(grid[1]), "B", tr("south → north"), c.ys[order] - c.cy,
                          c.terrain[order, col], None if base is None else base[order, col], c.inside[order, col],
                          colors)
        self.canvasProfiles.draw()

        faces, colors, z_values = self._surfaceFaces(1, exaggeration, stock_mode, base)
        self._fineMesh = self._coarseMesh = None
        if faces is not None:
            # one collection: the faces of the stock and of the base are depth-sorted together
            self._fineMesh = _fast_mesh(faces, colors)
            self.ax.add_collection3d(self._fineMesh)
            stride = int(math.ceil(c.xs.size / float(ROTATION_DISPLAY_NODES)))
            if stride > 1:
                coarse_faces, coarse_colors, _ = self._surfaceFaces(stride, exaggeration, stock_mode, base)
                if coarse_faces is not None:
                    self._coarseMesh = _fast_mesh(coarse_faces, coarse_colors)
                    self._coarseMesh.set_visible(False)
                    self.ax.add_collection3d(self._coarseMesh)
        for px, py, pz in c.outlines:
            self.ax.plot(px - c.cx, py - c.cy, pz, color="black", linewidth=1.5)
            z_values.append(pz)
        if self.currentMethod() == BaseLevelOptions.SPLINE_RING:   # ground read by the spline
            for px, py, pz in c.ringOutlines(self.spinRing.value()):
                self.ax.plot(px - c.cx, py - c.cy, pz, color="black", linewidth=1.2, linestyle=":")
        # traces of the cross-sections, slightly above the DEM so they stay visible
        lift = 0.02 * max(float(np.nanmax(c.terrain) - np.nanmin(c.terrain)), 0.1)
        self.ax.plot(c.xs - c.cx, np.full(c.xs.size, c.ys[row] - c.cy), c.terrain[row, :] + lift,
                     color=profile_colors["A"], linestyle="--", linewidth=1.4)
        self.ax.plot(np.full(c.ys.size, c.xs[col] - c.cx), c.ys - c.cy, c.terrain[:, col] + lift,
                     color=profile_colors["B"], linestyle="--", linewidth=1.4)
        for name, (x0, y0, z0, x1, y1, z1) in (
                ("A", (c.xs[0], c.ys[row], c.terrain[row, 0], c.xs[-1], c.ys[row], c.terrain[row, -1])),
                ("B", (c.xs[col], c.ys[-1], c.terrain[-1, col], c.xs[col], c.ys[0], c.terrain[0, col]))):
            for label, x, y, z in ((name, x0, y0, z0), (name + "'", x1, y1, z1)):
                if np.isfinite(z):
                    self.ax.text(x - c.cx, y - c.cy, z + 4 * lift, label, color=profile_colors[name],
                                 fontweight="bold")
        self._finishAxes(z_values, exaggeration, elev, azim)

    def _surfaceFaces(self, stride, exaggeration, stock_mode, base):
        """Quads (n, 4, 3), their RGBA colours and their heights, on every `stride`-th node."""
        c = self.computer
        xs, ys = c.xs[::stride], c.ys[::stride]
        gx, gy = np.meshgrid(xs - c.cx, ys - c.cy)
        qx = _quad_corners(gx).reshape(-1, 4)
        qy = _quad_corners(gy).reshape(-1, 4)
        qt = _quad_corners(c.terrain[::stride, ::stride]).reshape(-1, 4)
        inside_count = _quad_corners(c.inside[::stride, ::stride].astype(np.int8)).reshape(-1, 4).sum(axis=1)
        faces = []
        colors = []
        z_values = []

        # ground and stock surface
        valid = np.isfinite(qt).all(axis=1)
        in_stock = inside_count >= 2
        keep = valid & ~(in_stock & (stock_mode == STOCK_HIDDEN))
        if keep.any():
            shade = _shade(qx[keep], qy[keep], qt[keep], exaggeration)
            rgb = np.where(in_stock[keep][:, None], np.array([0.80, 0.62, 0.38]), GROUND_COLOR) * shade[:, None]
            alpha = np.where(in_stock[keep] & (stock_mode == STOCK_TRANSPARENT), 0.35, 1.0)
            faces.append(np.stack([qx[keep], qy[keep], qt[keep]], axis=-1))
            colors.append(np.column_stack([rgb, alpha]))
            z_values.append(qt[keep].ravel())

        # base surface, under the polygon only
        if base is not None:
            qb = _quad_corners(base[::stride, ::stride]).reshape(-1, 4)
            keep = (inside_count >= 1) & np.isfinite(qb).all(axis=1)
            if keep.any():
                with np.errstate(invalid="ignore"):
                    dz = np.nanmean(np.where(np.isfinite(qt[keep]), qt[keep] - qb[keep], np.nan), axis=1)
                above = np.nan_to_num(dz, nan=0.0) < 0
                rgba = np.where(above[:, None], np.array(BASE_ABOVE_GROUND), np.array(BASE_BELOW_GROUND))
                rgba[:, :3] = _lit(rgba[:, :3], qx[keep], qy[keep], qb[keep], exaggeration)
                faces.append(np.stack([qx[keep], qy[keep], qb[keep]], axis=-1))
                colors.append(rgba)
                z_values.append(qb[keep].ravel())

        if not faces:
            return None, None, z_values
        return np.concatenate(faces), np.concatenate(colors), z_values

    def _finishAxes(self, z_values, exaggeration, elev, azim):
        c = self.computer
        z = np.concatenate(z_values) if z_values else np.array([0.0, 1.0])
        z = z[np.isfinite(z)]
        z_lo, z_hi = (float(z.min()), float(z.max())) if z.size else (0.0, 1.0)
        z_range = max(z_hi - z_lo, 0.01)
        x_range = float(c.xs[-1] - c.xs[0]) or 1.0
        y_range = float(c.ys[0] - c.ys[-1]) or 1.0
        self.ax.set_xlim(c.xs[0] - c.cx, c.xs[-1] - c.cx)
        self.ax.set_ylim(c.ys[-1] - c.cy, c.ys[0] - c.cy)
        self.ax.set_zlim(z_lo, z_lo + z_range)
        try:
            self.ax.set_box_aspect((x_range, y_range, z_range * exaggeration), zoom=0.9)   # room for the near side, enlarged by the perspective
        except TypeError:   # matplotlib < 3.6
            self.ax.set_box_aspect((x_range, y_range, z_range * exaggeration))
        unit = self.dem_unit
        self.ax.set_xlabel("X ({})".format(unit))
        self.ax.set_ylabel("Y ({})".format(unit))
        self.ax.set_zlabel("Z ({})".format(unit))
        self.ax.view_init(elev=elev, azim=azim)
        self._drawn = True
        # synchronous: a deferred draw_idle() could fire after the window is closed (deleted canvas)
        self.canvas.draw()

    def _style3d(self, colors):
        """Light panes and grid, small muted ticks, few height ticks (they overlapped)."""
        from matplotlib.ticker import MaxNLocator
        pane = theme.mix(colors["card"], colors["text"], 0.035)
        pane = (pane.redF(), pane.greenF(), pane.blueF(), 1.0)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.set_facecolor(pane)
            axis.pane.set_edgecolor(colors["border"])
            axis.line.set_color(colors["border"])
            axis.set_major_locator(MaxNLocator(nbins=4 if axis is self.ax.zaxis else 6, steps=[1, 2, 5, 10]))
        for name in ("x", "y", "z"):
            self.ax.tick_params(axis=name, colors=colors["muted"], labelsize=7.5, pad=1)

    def _drawProfile(self, ax, name, direction, distance, terrain, base, inside, colors):
        """Cross-section through the centre of the polygon: DEM line, base under the polygon,
        filled in blue where the base is under the DEM and in red where it is above."""
        unit = self.dem_unit
        color = (PROFILE_COLORS_DARK if colors["dark"] else PROFILE_COLORS)[name]
        ax.plot(distance, terrain, color=theme.mix(colors["text"], colors["card"], 0.2).name(), linewidth=1.3,
                label=tr("DEM (ground and stock)"))
        if base is not None:
            under = np.where(inside, base, np.nan)
            ax.plot(distance, under, color=BASE_BELOW_GROUND[:3], linewidth=2.0, label=tr("base"))
            with np.errstate(invalid="ignore"):
                ax.fill_between(distance, under, terrain, where=inside & (terrain >= under),
                                color=BASE_BELOW_GROUND[:3], alpha=0.25, interpolate=True, linewidth=0)
                ax.fill_between(distance, under, terrain, where=inside & (terrain < under),
                                color=BASE_ABOVE_GROUND[:3], alpha=0.45, interpolate=True, linewidth=0)
        edges = np.nonzero(np.diff(inside.astype(np.int8)))[0]
        for i in edges:
            ax.axvline(0.5 * (distance[i] + distance[i + 1]), color=colors["muted"], linewidth=0.8, linestyle=":")
        ax.set_title(tr("Section {n}–{n}' ({d})").format(n=name, d=direction), fontsize=9, color=color,
                     fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(distance[0], distance[-1])
        # same vertical exaggeration as the 3D view (1 = true scale); the height range is
        # enlarged to fill the frame instead of squashing the profile
        exaggeration = self.spinExaggeration.value()
        ax.set_aspect(exaggeration, adjustable="datalim")
        ax.set_xlabel(tr("distance from centre ({})").format(unit))
        ax.set_ylabel(tr("height ({})").format(unit))
        ax.tick_params(length=3)
        ax.grid(True)
        ax.text(0.0, 1.0, name, transform=ax.transAxes, color=color, fontweight="bold", va="bottom")
        ax.text(1.0, 1.0, name + "'", transform=ax.transAxes, color=color, fontweight="bold", va="bottom",
                ha="right")
        if name == "A":
            ax.legend(loc="upper right", frameon=False)
