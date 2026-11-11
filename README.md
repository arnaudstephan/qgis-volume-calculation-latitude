# Volume Calculation Tool - Latitude Drone (QGIS plugin)

QGIS plugin computing stockpile volumes inside polygons from a DEM or a LAS/LAZ ground point cloud,
with a per-polygon base surface and a 3D view of each stock and its base.

- Plugin source and user guide: [volume_calculation_tool/](volume_calculation_tool/README.md)
- Build the zip for the QGIS plugin repository: `python build_plugin_zip.py` (with the Python of QGIS)
  → `dist/volume_calculation_latitude_<version>.zip`
- Tests: `python-qgis -m unittest discover -s volume_calculation_tool/test`

Based on the [Volume Calculation Tool](https://github.com/REDcatch/Volume_calculation_for_QGIS3) by
REDcatch GmbH. License: GNU GPL v3, see [LICENSE](LICENSE).
