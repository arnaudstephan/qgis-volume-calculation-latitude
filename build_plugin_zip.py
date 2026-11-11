"""Build the zip of the plugin for the QGIS plugin repository and check its rules.

Usage: python build_plugin_zip.py  ->  dist/volume_calculation_latitude_<version>.zip

Only the runtime files listed below go into the zip, under one root folder named after the
plugin package. Tests, translation sources and developer tools stay out.
"""
import configparser
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "volume_calculation_tool")
PACKAGE = "volume_calculation_latitude"      # folder name of the installed plugin
RUNTIME_FILES = [
    "__init__.py",
    "base_preview.py",
    "theme.py",
    "translation.py",
    "volume_calculation_tool.py",
    "volume_calculation_tool_dialog.py",
    "volume_calculation_tool_dialog_base.ui",
    "volume_engine.py",
    "ndom_default.qml",
    "icon.png",
    "icon.svg",
    "i18n/VolumeCalculationTool_fr.qm",
    "metadata.txt",
    "LICENSE",
    "README.md",
]
REQUIRED_METADATA = ["name", "qgisMinimumVersion", "description", "about", "version", "author", "email",
                     "repository"]
MAX_SIZE = 25 * 1024 * 1024
FORBIDDEN = re.compile(r"(^|/)(__pycache__|\.git|__MACOSX)(/|$)|\.pyc$|(^|/)(ui_[^/]*|resources_rc|resources)\.py$")


def check(condition, message, errors):
    if not condition:
        errors.append(message)


def main():
    errors = []
    check(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", PACKAGE) is not None,
          "package name must be ASCII letters, digits, '-' or '_' and not start with a digit", errors)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(os.path.join(SOURCE, "metadata.txt"), encoding="utf-8")
    meta = parser["general"] if parser.has_section("general") else {}
    for key in REQUIRED_METADATA:
        check(bool(meta.get(key, "").strip()), "metadata.txt: '{}' is missing".format(key), errors)
    version = meta.get("version", "").strip()
    check(re.fullmatch(r"\d+(\.\d+){0,2}", version) is not None,
          "metadata.txt: version '{}' is not like 1.2 or 1.2.3".format(version), errors)
    check(version in meta.get("changelog", ""), "metadata.txt: no changelog entry for {}".format(version), errors)
    check(os.path.isfile(os.path.join(SOURCE, meta.get("icon", "icon.png"))), "icon file missing", errors)

    for name in RUNTIME_FILES:
        check(os.path.isfile(os.path.join(SOURCE, name)), "missing file: " + name, errors)
        check(not FORBIDDEN.search(name), "generated or hidden file listed: " + name, errors)
    if errors:
        sys.exit("Not built:\n  " + "\n  ".join(errors))

    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    out = os.path.join(HERE, "dist", "{}_{}.zip".format(PACKAGE, version))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in RUNTIME_FILES:
            archive.write(os.path.join(SOURCE, name), PACKAGE + "/" + name)
    size = os.path.getsize(out)
    if size > MAX_SIZE:
        os.remove(out)
        sys.exit("Not built: the zip is {:.1f} MB, the repository accepts 25 MB".format(size / 1048576))
    print("{} ({:.0f} kB)".format(out, size / 1024))


if __name__ == "__main__":
    main()
