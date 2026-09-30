"""
Assemble the thesis notebook from the ``py:percent`` source parts.

Keeping the notebook's source as plain ``.py`` files means it diffs and reviews
like code; this script turns them into the ``.ipynb`` that is actually run.

    python src/build_notebook.py            # build only
    python src/build_notebook.py --execute  # build, then run it and embed outputs
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["thesis_pipeline_part1.py", "thesis_pipeline_part2.py", "thesis_pipeline_part3.py"]
OUT = ROOT / "notebooks" / "Ghana_Petroleum_Price_Forecasting.ipynb"

CELL_RE = re.compile(r"^# %%(.*)$")


def parse_percent(text: str):
    """Split a py:percent script into (kind, source) cells."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if src.strip():
            cells.append((kind, src))

    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def strip_markdown(src: str) -> str:
    """Markdown cells are written as comments; take the comment markers off."""
    out = []
    for line in src.splitlines():
        out.append(line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line))
    return "\n".join(out).strip("\n")


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for part in PARTS:
        text = (ROOT / "notebooks" / part).read_text(encoding="utf-8")
        for kind, src in parse_percent(text):
            if kind == "markdown":
                nb.cells.append(nbf.v4.new_markdown_cell(strip_markdown(src)))
            else:
                nb.cells.append(nbf.v4.new_code_cell(src))

    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["title"] = "Forecasting Petroleum Product Prices in Ghana"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


def execute(path: Path, timeout: int = 5400) -> None:
    from nbclient import NotebookClient

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}},
                            allow_errors=False)
    client.execute()
    nbf.write(nb, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    out = build()
    n_code = sum(c.cell_type == "code" for c in nbf.read(out, as_version=4).cells)
    print("built %s  (%d cells, %d code)" % (out, len(nbf.read(out, as_version=4).cells), n_code))
    if args.execute:
        print("executing ...")
        execute(out)
        print("executed and saved with outputs")
