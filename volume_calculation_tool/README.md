# Volume Calculation Tool For QGIS

The volume calculation tool is a plugin for QGIS which allows for easy calculation of volumes defined by polygons and DEM layers.

## Installation

- Use the internal QGIS plugin manager, or *Plugins › Manage and Install Plugins › Install from ZIP*.
- To build the zip for the QGIS plugin repository, run `python build_plugin_zip.py` in the folder
  above the plugin: it writes `dist/volume_calculation_latitude_<version>.zip` (root folder
  `volume_calculation_latitude`, runtime files only, version read from `metadata.txt`).

## Usage

1. Select the polygon layer and the height layer: a DEM raster (and band) **or a LAS/LAZ point cloud
   of ground points**. A point cloud is triangulated (TIN, PDAL tools of QGIS >= 3.32) into a DEM at the
   chosen resolution, only over the included polygons; optionally only class 2 (ground) points are used,
   and the interpolated DEM can be saved next to the LAS file and added to the project.
2. In the polygon table ("Polygons to compute"), check the polygons to compute and choose for
   **each polygon** its base method. The calculation only uses the method of each row. To change
   several rows at once, select them and change one of them, right-click, or use *Set base method*
   (selected, checked or all rows):
   - *Outline – lowest point* / *Outline – average*: horizontal base from the DEM heights along the whole polygon outline
   - *Outline – best-fit plane*: inclined base fitted to the outline heights (stockpiles on sloped ground)
   - *Outline – triangulated (TIN)*: Delaunay TIN of the DEM heights at the polygon vertices only (as Pix4D
     "Triangulated"): place the vertices on the natural ground; the outline between them is not sampled
   - *Spline on ground ring*: smooth curved surface (thin-plate spline) fitted on a band of ground around
     the polygon (other checked polygons excluded); extends valleys, bulges and slope changes under the
     pile. Ring width and stiffness (0 supple … 10 nearly a plane) are set for each polygon in the 3D view
     and stored in `Base_ring` / `Base_stiff`. The fit RMS residual is logged as a quality indicator.
   - *Base DEM (surface to surface)*: compared with the base DEM layer chosen in *Settings of the base
     methods* (reprojected on the fly if needed)
   - *Manual level*: constant level, asked when the method is chosen (a missing level is shown in red
     and refused by the calculation)
3. Click *Compute volumes*. The volumes are shown in the table and the log, and written to the
   `V_above` / `V_below` fields (m³ by default, yd³ or ft³ optional), the method used to `Base_meth`
   and the base height to `Base_z`. The choices are read back from these fields the next time, and
   the polygons can be labeled with their volumes. Double-click a row to zoom the map on the polygon.

Notes:
- The default sampling step is the DEM pixel size; the grid is aligned on the DEM pixels.
- Nodata cells are skipped and a coverage warning is logged when part of a polygon has no DEM data.
- The DEM must be in a projected CRS (metres or feet); heights are assumed in the same unit.

See also the documentation in the help folder and/or the tutorial video in the example folder.

## Tests

Run with the Python of a QGIS installation, from the folder containing `volume_calculation_tool`:

    python-qgis volume_calculation_tool/test/test_volume_engine.py
    python-qgis volume_calculation_tool/test/test_plugin_gui.py
    python-qgis volume_calculation_tool/test/test_volume_verification.py
    python-qgis volume_calculation_tool/test/test_translation.py

`test_volume_verification.py` checks the engine against code it does not share: an
independent numpy reference implementation on ~500 randomised scenes (volumes and cell
counts must agree to 1e-9), closed-form volumes (cone, paraboloid, spherical cap, frustum),
QGIS's own "Raster surface volume" algorithm, invariances (split polygons, vertex order,
translation), units (international / US survey feet), scale/offset DEMs and the nDOM.

## Translations

The plugin follows the language of the QGIS interface: French when QGIS is in French
(`i18n/VolumeCalculationTool_fr.qm`), English otherwise. Source texts are in English; in the code
they go through `tr("...")` (or `tr_noop("...")` for module constants), see `translation.py`.

After changing or adding texts (plain Python 3, no Qt tools needed):

    python i18n/i18n_tool.py update     # adds the new texts to i18n/VolumeCalculationTool_fr.ts
    # translate the entries marked type="unfinished" (Qt Linguist or any text editor)
    python i18n/i18n_tool.py check      # missing translations, placeholders {} / HTML / \n / % differing
    python i18n/i18n_tool.py build      # compiles the .qm loaded by the plugin

`test_translation.py` fails when a text is not translated or the .qm is older than the .ts.

## Build With

[Qgis-Plugin Builder](https://github.com/g-sherman/Qgis-Plugin-Builder/)

## License
Copyright (C) 2020-2026 REDcatch GmbH (original plugin).
Modifications (C) 2026 Latitude Drone (per-polygon base level, new calculation engine, labels, ...).
[GPL](https://www.gnu.org/licenses/gpl-3.0.en.html)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
