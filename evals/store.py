"""Where the evaluation data lives: the Queries and Labels sheets of evals/eval_results.xlsx.

The workbook is the only copy of the 20 test products and the judge labels. Every script reads
them from it. run_eval.py rebuilds the workbook on each run and writes these two sheets back as
they were read, so a label changed in Excel is kept.

    Queries  one row per test product: query_id, kind (random / hard), why it was chosen, the product
    Labels   one row per judged candidate: good (yes / no), gt_rank (1 to 10 for the ground truth),
             source (shortlist, sweep, or the method whose result it was)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill

WORKBOOK_PATH = Path("evals/eval_results.xlsx")
QUERY_COLUMNS = ["query_id", "kind", "why", "uniq_id", "name", "category", "gender", "price"]
LABEL_COLUMNS = ["query_id", "position", "uniq_id", "name", "category", "gender", "price", "brand", "colours",
                 "good", "gt_rank", "source"]
EDITABLE = {"good", "gt_rank"}
WIDTHS = {"query_id": 9, "kind": 8, "why": 46, "uniq_id": 34, "name": 70, "category": 24, "gender": 8, "price": 9,
          "position": 8, "brand": 18, "colours": 18, "good": 7, "gt_rank": 8, "source": 10}
HEADER = Font(name="Arial", size=10, bold=True, color="FFFFFF")
BODY = Font(name="Arial", size=10)
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
EDIT_FILL = PatternFill("solid", fgColor="FFFF00")
PRICE_FORMAT = '"₹"#,##0;-"₹"#,##0;"-"'


def has_data(path: Path = WORKBOOK_PATH) -> bool:
    return path.exists() and {"Queries", "Labels"} <= set(load_workbook(path, read_only=True).sheetnames)


def load_data(path: Path = WORKBOOK_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The test products and the labels, as two tables."""
    sheets = pd.read_excel(path, sheet_name=["Queries", "Labels"],
                           dtype={"query_id": str, "uniq_id": str, "good": str, "source": str})
    queries = sheets["Queries"][QUERY_COLUMNS]
    labels = sheets["Labels"][LABEL_COLUMNS].copy()
    labels["gt_rank"] = pd.to_numeric(labels["gt_rank"], errors="coerce")
    return queries, labels


def _sheet(wb: Workbook, title: str, table: pd.DataFrame, columns: list[str]) -> None:
    ws = wb.create_sheet(title)
    for c, column in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=c, value=column)
        cell.font, cell.fill = HEADER, HEADER_FILL
        ws.column_dimensions[cell.column_letter].width = WIDTHS.get(column, 12)
    for r, row in enumerate(table[columns].itertuples(index=False), start=2):
        for c, (column, value) in enumerate(zip(columns, row), start=1):
            cell = ws.cell(row=r, column=c, value=None if pd.isna(value) or value == "" else value)
            cell.font = BODY
            if column == "price":
                cell.number_format = PRICE_FORMAT
            if column in EDITABLE:
                cell.fill = EDIT_FILL
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def write_data_sheets(wb: Workbook, queries: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Add the Queries and Labels sheets to a workbook (yellow cells are the ones to edit)."""
    _sheet(wb, "Queries", queries, QUERY_COLUMNS)
    _sheet(wb, "Labels", labels, LABEL_COLUMNS)


def save(wb: Workbook, path: Path = WORKBOOK_PATH) -> None:
    try:
        wb.save(path)
    except PermissionError:
        raise SystemExit(f"Cannot write {path}: it is probably open in Excel. Close it and run again.")


def save_data(queries: pd.DataFrame, labels: pd.DataFrame, path: Path = WORKBOOK_PATH) -> None:
    """Replace only the Queries and Labels sheets, keeping any other sheets (used by shortlist and sweep)."""
    if path.exists():
        wb = load_workbook(path)
        for title in ("Queries", "Labels"):
            if title in wb.sheetnames:
                del wb[title]
    else:
        wb = Workbook()
        wb.remove(wb.active)
    write_data_sheets(wb, queries, labels)
    save(wb, path)
