from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PRESSUREMAT_GLOB = "*_pressuremat.json"


def parse_number(value: str) -> Optional[Any]:
    """Convert numeric strings to int/float, 'n/a'/blank to None, keep others."""
    stripped = value.strip()
    if not stripped or stripped.lower() == "n/a":
        return None

    int_pattern = re.compile(r"^-?\d+$")
    float_pattern = re.compile(r"^-?\d+\.\d+$")

    if int_pattern.match(stripped):
        return int(stripped)
    if float_pattern.match(stripped):
        return float(stripped)

    # Fallback: attempt generic float conversion (covers scientific notation)
    try:
        return float(stripped)
    except ValueError:
        return stripped


def next_non_empty(rows: List[List[str]], start: int) -> int:
    """Advance the index to the next non-empty row."""
    idx = start
    while idx < len(rows) and all(cell.strip() == "" for cell in rows[idx]):
        idx += 1
    return idx


def trim_row(row: List[str]) -> List[str]:
    """Strip whitespace and drop trailing empty cells."""
    trimmed = [cell.strip() for cell in row]
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def parse_gait_table(rows: List[List[str]], start: int) -> tuple[Dict[str, Any], int]:
    """Parse the gait table section starting at index start."""
    idx = next_non_empty(rows, start)
    header = trim_row(rows[idx])
    if not header or header[0] != "Gait Table":
        raise ValueError("Expected 'Gait Table' header")
    label = header[1] if len(header) > 1 else ""
    idx += 1

    metrics: Dict[str, Any] = {}
    idx = next_non_empty(rows, idx)
    while idx < len(rows) and rows[idx] and rows[idx][0].strip():
        row = trim_row(rows[idx])
        if not row:
            break
        key = row[0]
        value = parse_number(row[1]) if len(row) > 1 else None
        metrics[key] = value
        idx += 1
    return {"label": label, "metrics": metrics}, idx


def parse_stance_stride_table(rows: List[List[str]], start: int) -> tuple[Dict[str, Any], int]:
    """Parse the stance-stride table section starting at index start."""
    idx = next_non_empty(rows, start)
    header = trim_row(rows[idx])
    if not header or header[0] != "Stance-Stride Table":
        raise ValueError("Expected 'Stance-Stride Table' header")
    label = header[1] if len(header) > 1 else ""
    idx += 1

    idx = next_non_empty(rows, idx)
    column_row = trim_row(rows[idx])
    if not column_row or column_row[0] != "":
        raise ValueError("Expected column header row for stance-stride table")
    columns = [cell for cell in column_row[1:] if cell != ""]
    idx += 1

    rows_data: List[Dict[str, Any]] = []
    idx = next_non_empty(rows, idx)
    while idx < len(rows) and rows[idx] and rows[idx][0].strip():
        row = trim_row(rows[idx])
        metric_name = row[0]
        values = row[1:]
        entry: Dict[str, Any] = {"metric": metric_name}
        for col, raw_value in zip(columns, values):
            entry[col] = parse_number(raw_value)
        rows_data.append(entry)
        idx += 1

    return {"label": label, "columns": columns, "rows": rows_data}, idx


SYMMETRY_SECTION_PATTERN = re.compile(
    r"^(?P<metric>.+?) (?P<section>Front / Hind|Left / Right|Left Front / Right Front|Left Hind / Right Hind)$"
)


def parse_symmetry_table(rows: List[List[str]], start: int) -> Dict[str, Any]:
    """Parse the symmetry table section starting at index start."""
    idx = next_non_empty(rows, start)
    header = trim_row(rows[idx])
    if not header or header[0] != "Symmetry Table":
        raise ValueError("Expected 'Symmetry Table' header")
    label = header[1] if len(header) > 1 else ""
    idx += 1

    idx = next_non_empty(rows, idx)
    sections: Dict[str, Dict[str, Any]] = {}
    while idx < len(rows):
        row = trim_row(rows[idx])
        idx += 1
        if not row:
            continue

        metric_text = row[0]
        value = parse_number(row[1]) if len(row) > 1 else None

        match = SYMMETRY_SECTION_PATTERN.match(metric_text)
        if match:
            metric = match.group("metric")
            section = match.group("section")
            sections.setdefault(section, {})[metric] = value
        else:
            sections.setdefault("Other", {})[metric_text] = value

    return {"label": label, "sections": sections}


def convert_file(path: Path) -> None:
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader]

    gait_table, idx_after_gait = parse_gait_table(rows, 0)
    stance_stride_table, idx_after_stance = parse_stance_stride_table(rows, idx_after_gait)
    symmetry_table = parse_symmetry_table(rows, idx_after_stance)

    structured = {
        "gait_table": gait_table,
        "stance_stride_table": stance_stride_table,
        "symmetry_table": symmetry_table,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main() -> None:
    videos_dir = Path("videos")
    files = sorted(videos_dir.glob(PRESSUREMAT_GLOB))
    if not files:
        raise SystemExit("No pressuremat JSON files found.")

    for file_path in files:
        convert_file(file_path)


if __name__ == "__main__":
    main()
