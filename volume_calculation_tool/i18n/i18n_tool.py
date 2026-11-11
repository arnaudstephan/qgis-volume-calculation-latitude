# -*- coding: utf-8 -*-
"""
Copyright (C) 2026 Latitude Drone - GNU GPL v2 or later (see LICENSE).
Translation files of the plugin (development tool, plain Python 3, no Qt tools needed).

    python i18n_tool.py update [translations.json ...]   extract the texts of the code and of the .ui
                                                         into VolumeCalculationTool_<lang>.ts (keeps the
                                                         existing translations; a JSON file
                                                         {"context": {"source": "translation"}} adds some)
    python i18n_tool.py check                            missing / obsolete translations, placeholders
    python i18n_tool.py build                            .ts -> .qm (what the plugin loads)

Texts of the code: literal first argument of tr(...), self.tr(...) and tr_noop(...); tr(CONSTANT) is
allowed when CONSTANT is defined with tr_noop("...") (an f-string is refused: it cannot be translated).
Texts of the .ui: every <string> without notr="true", in the context of the form class.
lrelease (Qt Linguist) gives the same .qm; this script avoids needing it.
"""

import ast
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(HERE)
PY_CONTEXT = "VolumeCalculationTool"
LANGUAGES = {"fr": "fr_FR"}
TS_PATTERN = "VolumeCalculationTool_{}.ts"
QM_PATTERN = "VolumeCalculationTool_{}.qm"
FUNCTIONS = {"tr", "tr_noop"}
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


# ---------------------------------------------------------------------- extraction
def python_texts(path):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), path)
    texts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name not in FUNCTIONS:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            texts.append((arg.lineno, arg.value))
        elif name == "tr_noop" or isinstance(arg, ast.JoinedStr):
            raise SystemExit("{}:{}: {}() needs a literal text (no f-string / variable)".format(
                path, node.lineno, name))
        # tr(CONSTANT): the text of the constant is extracted where it is defined with tr_noop()
    return [text for _, text in sorted(texts)]


def ui_texts(path):
    root = ET.parse(path).getroot()
    context = root.findtext("class")
    texts = []
    for element in root.iter("string"):
        if element.get("notr") == "true" or not (element.text or "").strip():
            continue
        texts.append(element.text)
    return context, texts


def extract():
    """{context: [source, ...]} in the order of the files."""
    contexts = {}
    for name in sorted(os.listdir(PLUGIN_DIR)):
        path = os.path.join(PLUGIN_DIR, name)
        if name.endswith(".py") and name != "resources.py":
            for text in python_texts(path):
                contexts.setdefault(PY_CONTEXT, [])
                if text not in contexts[PY_CONTEXT]:
                    contexts[PY_CONTEXT].append(text)
        elif name.endswith(".ui"):
            context, texts = ui_texts(path)
            for text in texts:
                contexts.setdefault(context, [])
                if text not in contexts[context]:
                    contexts[context].append(text)
    return contexts


# ---------------------------------------------------------------------- .ts
def read_ts(path):
    """{context: {source: translation or None}}"""
    out = {}
    if not os.path.exists(path):
        return out
    for context in ET.parse(path).getroot().iter("context"):
        name = context.findtext("name")
        for message in context.iter("message"):
            translation = message.find("translation")
            text = None
            if translation is not None and translation.get("type") not in ("unfinished", "obsolete", "vanished"):
                text = translation.text or ""
            out.setdefault(name, {})[message.findtext("source")] = text
    return out


def write_ts(path, language, contexts, translations):
    root = ET.Element("TS", {"version": "2.1", "language": LANGUAGES[language], "sourcelanguage": "en"})
    for name, sources in contexts.items():
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = name
        for source in sources:
            message = ET.SubElement(context, "message")
            ET.SubElement(message, "source").text = source
            text = translations.get(name, {}).get(source)
            translation = ET.SubElement(message, "translation")
            if text is None:
                translation.set("type", "unfinished")
            else:
                translation.text = text
    ET.indent(root, space="    ")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n')
        handle.write(ET.tostring(root, encoding="unicode"))
        handle.write("\n")


# ---------------------------------------------------------------------- .qm (format of Qt's QTranslator)
QM_MAGIC = bytes([0x3C, 0xB8, 0x64, 0x18, 0xCA, 0xEF, 0x9C, 0x95, 0xCD, 0x21, 0x1C, 0xBF, 0x60, 0xA1, 0xBD, 0xDD])
TAG_HASHES, TAG_MESSAGES, TAG_LANGUAGE = 0x42, 0x69, 0xA7
TAG_END, TAG_TRANSLATION, TAG_SOURCE, TAG_CONTEXT = 1, 3, 6, 7


def elf_hash(data):
    h = 0
    for byte in data:
        h = ((h << 4) + byte) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h or 1


def write_qm(path, language, translations):
    messages = bytearray()
    hashes = []
    for context, entries in translations.items():
        for source, text in entries.items():
            if text is None or text == "":
                continue
            source_bytes = source.encode("utf-8")
            translated = text.encode("utf-16-be")
            hashes.append((elf_hash(source_bytes), len(messages)))
            messages += bytes([TAG_TRANSLATION]) + struct.pack(">I", len(translated)) + translated
            messages += bytes([TAG_SOURCE]) + struct.pack(">I", len(source_bytes)) + source_bytes
            context_bytes = context.encode("utf-8")
            messages += bytes([TAG_CONTEXT]) + struct.pack(">I", len(context_bytes)) + context_bytes
            messages += bytes([TAG_END])
    hashes.sort()
    hash_table = b"".join(struct.pack(">II", h, offset) for h, offset in hashes)
    lang = LANGUAGES[language].encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(QM_MAGIC)
        handle.write(bytes([TAG_LANGUAGE]) + struct.pack(">I", len(lang)) + lang)
        handle.write(bytes([TAG_HASHES]) + struct.pack(">I", len(hash_table)) + hash_table)
        handle.write(bytes([TAG_MESSAGES]) + struct.pack(">I", len(messages)) + bytes(messages))
    return len(hashes)


# ---------------------------------------------------------------------- checks
def problems(contexts, translations):
    out = []
    for name, sources in contexts.items():
        for source in sources:
            text = translations.get(name, {}).get(source)
            if text is None:
                out.append("missing [{}]: {!r}".format(name, source))
                continue
            if PLACEHOLDER.findall(source) != PLACEHOLDER.findall(text):
                out.append("placeholders differ [{}]: {!r} -> {!r}".format(name, source, text))
            for mark in ("<", "\n", "%"):
                if source.count(mark) != text.count(mark):
                    out.append("{!r} count differs [{}]: {!r} -> {!r}".format(mark, name, source, text))
    for name, entries in translations.items():
        for source in entries:
            if source not in contexts.get(name, []):
                out.append("obsolete [{}]: {!r}".format(name, source))
    return out


def main(argv):
    if len(argv) < 2 or argv[1] not in ("update", "check", "build"):
        print(__doc__)
        return 2
    command = argv[1]
    status = 0
    for language in LANGUAGES:
        ts_path = os.path.join(HERE, TS_PATTERN.format(language))
        translations = read_ts(ts_path)
        contexts = extract()
        if command == "update":
            for json_path in argv[2:]:
                with open(json_path, encoding="utf-8") as handle:
                    added = json.load(handle)
                for name, entries in added.items():
                    translations.setdefault(name, {}).update(entries)
            kept = {n: {s: t for s, t in e.items() if s in contexts.get(n, [])} for n, e in translations.items()}
            write_ts(ts_path, language, contexts, kept)
            total = sum(len(s) for s in contexts.values())
            done = sum(1 for n, s in contexts.items() for x in s if kept.get(n, {}).get(x) is not None)
            print("{}: {} texts, {} translated".format(os.path.basename(ts_path), total, done))
        elif command == "check":
            found = problems(contexts, translations)
            for line in found:
                print(line)
            print("{}: {} problem(s)".format(language, len(found)))
            status = 1 if found else status
        else:
            count = write_qm(os.path.join(HERE, QM_PATTERN.format(language)), language, translations)
            print("{}: {} translations compiled".format(QM_PATTERN.format(language), count))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
