"""Score similarity methods against the hand-labelled ground truth and write an Excel workbook.

Run from the repo root, after `python -m similarity.preprocessing`:

    python -m evals.run_eval                           # every method on the Methods sheet; tfidf in Details
    python -m evals.run_eval --alternative minilm      # plus a second method in Summary and Details

Everything lives in evals/eval_results.xlsx. The run reads the test products and hand labels
from its Queries and Labels sheets (see evals/store.py), then rebuilds the workbook:

    Methods  one row per method with its ranking metrics
    Summary  the metrics per test product, as formulas over Details, and the overall numbers
    Details  each method's top 10 beside the ground truth
    Queries  the 20 test products (written back as read)
    Labels   the hand labels (written back as read, so edits made in Excel are kept)

The same summary is printed in the terminal. A result nobody has labelled yet is added to the
Labels sheet with an empty `good` cell, shows as "not labelled" (never as wrong), and the run
prints a warning: fill in yes / no and run again.

A method whose files are missing (for example minilm before `python -m similarity.embeddings`)
is skipped with a message.

Three metrics per query, each over the method's top 10:
    good results        how many of the 10 were labelled good
    ground truth found  how many of the hand-picked best 10 the method also returned
    first good result   the rank of the first good result (1 is best)
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from evals.store import WORKBOOK_PATH, load_data, save, write_data_sheets
from similarity.features import genders
from similarity.preprocessing import load_products
from similarity.search import METHOD_WEIGHTS, load_features, rank

TOP_K = 10
NONE_RANK = TOP_K + 1   # "no good result" counts as rank 11 in the average

BASELINE = "tfidf"   # always shown in Summary and Details


def by_weights(weights: dict[str, float]):
    """A method that ranks with the stage 3 search and these weights."""
    return lambda features, row: rank(features, row, weights)[0]


# A method takes the features and a query row and returns candidate rows, best first.
# To compare a new method, add it here; it gets a row on the Methods sheet.
METHODS = {
    "tfidf": ("TF-IDF title + other attributes (stage 3, the API's method)", by_weights(METHOD_WEIGHTS["tfidf"])),
    "minilm": ("MiniLM title embedding + other attributes (stage 6)", by_weights(METHOD_WEIGHTS["minilm"])),
}


# --- the three metrics, for the terminal and the tests (the workbook uses formulas) ---------

def good_results(results: list[str], good: set[str]) -> int:
    return sum(uniq_id in good for uniq_id in results)


def ground_truth_found(results: list[str], truth: list[str]) -> int:
    return len(set(results) & set(truth))


def first_good(results: list[str], good: set[str]) -> int | None:
    return next((rank for rank, uniq_id in enumerate(results, start=1) if uniq_id in good), None)


def score(results: dict[str, list[str]], labels: pd.DataFrame) -> pd.DataFrame:
    """One row per query with the three metrics and the count of unlabelled results."""
    records = []
    for query_id, top in results.items():
        mine = labels[labels["query_id"] == query_id]
        good = set(mine.loc[mine["good"] == "yes", "uniq_id"])
        labelled = set(mine.loc[mine["good"].isin(["yes", "no"]), "uniq_id"])
        truth = mine.dropna(subset=["gt_rank"]).sort_values("gt_rank")["uniq_id"].tolist()
        records.append(
            {
                "query_id": query_id,
                "good_results": good_results(top, good),
                "ground_truth_found": ground_truth_found(top, truth),
                "ground_truth_size": len(truth),
                "first_good": first_good(top, good),
                "not_labelled": sum(u not in labelled for u in top),
            }
        )
    return pd.DataFrame(records).set_index("query_id")


def overall(table: pd.DataFrame) -> dict[str, float]:
    """The averages shown in the Summary sheet's bottom rows."""
    return {
        "good results (of 10)": table["good_results"].mean(),
        "ground truth found": table["ground_truth_found"].sum() / table["ground_truth_size"].sum(),
        "first good result": table["first_good"].fillna(NONE_RANK).mean(),
        "first result good": (table["first_good"] == 1).mean(),
    }


# --- ranking metrics for the Methods sheet: the hand-picked 10 are the right answers --------

RANKING_METRICS = {   # name: (number format, one-line meaning)
    "Recall@5": ("0%", "share of the hand picks found in the top 5 (at most 50% with 10 picks)"),
    "Recall@10": ("0%", "share of the hand picks found in the top 10"),
    "MRR@10": ("0.00", "1 / rank of the first hand pick, 0 if none in the top 10"),
    "Good@10": ("0%", "share of the top 10 labelled good (uses the good / not good labels)"),
}


def ranking_metrics(top: list[str], truth: list[str], good: set[str]) -> dict[str, float]:
    """The ranking metrics for one test product: `top` is the method's list, best first."""
    picks = set(truth)
    hit = [uniq_id in picks for uniq_id in top[:TOP_K]]
    return {
        "Recall@5": sum(hit[:5]) / len(truth),
        "Recall@10": sum(hit) / len(truth),
        "MRR@10": next((1 / r for r, h in enumerate(hit, start=1) if h), 0.0),
        "Good@10": sum(uniq_id in good for uniq_id in top[:TOP_K]) / TOP_K,
    }


def method_scores(results: dict[str, list[str]], labels: pd.DataFrame) -> dict[str, float]:
    """Each ranking metric averaged over the test products."""
    rows = []
    for query_id, top in results.items():
        mine = labels[labels["query_id"] == query_id]
        truth = mine.dropna(subset=["gt_rank"]).sort_values("gt_rank")["uniq_id"].tolist()
        rows.append(ranking_metrics(top, truth, set(mine.loc[mine["good"] == "yes", "uniq_id"])))
    return pd.DataFrame(rows).mean().to_dict()


# --- workbook ---------------------------------------------------------------------------

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
GROUP_FILL = PatternFill("solid", fgColor="D9E1F2")
YES_FILL = PatternFill("solid", fgColor="C6EFCE")
NO_FILL = PatternFill("solid", fgColor="FFC7CE")
UNKNOWN_FILL = PatternFill("solid", fgColor="FFEB9C")
PRICE_FORMAT = '"₹"#,##0;-"₹"#,##0;"-"'
NOT_LABELLED = "not labelled"


def _header(ws, row: int, titles: list[str]) -> None:
    for column, title in enumerate(titles, start=1):
        cell = ws.cell(row=row, column=column, value=title)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def _group(ws, row: int, first: int, last: int, title: str) -> None:
    ws.merge_cells(start_row=row, start_column=first, end_row=row, end_column=last)
    cell = ws.cell(row=row, column=first, value=title)
    cell.font = Font(bold=True)
    cell.fill = GROUP_FILL


def _widths(ws, widths: dict[str, float]) -> None:
    for column, width in widths.items():
        ws.column_dimensions[column].width = width


def _yes_no_colours(ws, cell_range: str) -> None:
    ws.conditional_formatting.add(cell_range, CellIsRule(operator="equal", formula=['"yes"'], fill=YES_FILL))
    ws.conditional_formatting.add(cell_range, CellIsRule(operator="equal", formula=['"no"'], fill=NO_FILL))
    ws.conditional_formatting.add(cell_range, CellIsRule(operator="equal", formula=[f'"{NOT_LABELLED}"'], fill=UNKNOWN_FILL))


def _clean(value):
    return None if pd.isna(value) else value


def details_sheet(wb: Workbook, queries: pd.DataFrame, labels: pd.DataFrame, results: dict, info: pd.DataFrame,
                  methods: list[str]) -> None:
    """10 rows per test product: the ground truth beside each method's results, all plain values."""
    ws = wb.create_sheet("Details")
    _group(ws, 1, 1, 3, "Test product")
    _group(ws, 1, 4, 6, "Ground truth: the 10 most similar, picked by hand")
    titles = ["Query ID", "Query", "Rank", "uniq_id", "Name", "Price"]
    for i, method in enumerate(methods):
        first = 7 + 5 * i
        _group(ws, 1, first, first + 4, METHODS[method][0])
        titles += ["uniq_id", "Name", "Price", "Good?", "In ground truth?"]
    _header(ws, 2, titles)

    good_of = {(q, u): g for q, u, g in zip(labels["query_id"], labels["uniq_id"], labels["good"]) if g in ("yes", "no")}
    r = 3
    for query in queries.itertuples(index=False):
        mine = labels[labels["query_id"] == query.query_id]
        truth = mine.dropna(subset=["gt_rank"]).sort_values("gt_rank")
        truth_ids = set(truth["uniq_id"])
        for position in range(1, TOP_K + 1):
            ws.cell(row=r, column=1, value=query.query_id)
            ws.cell(row=r, column=2, value=query.name)
            ws.cell(row=r, column=3, value=position)
            if position <= len(truth):
                pick = truth.iloc[position - 1]
                ws.cell(row=r, column=4, value=pick["uniq_id"])
                ws.cell(row=r, column=5, value=_clean(pick["name"]))
                ws.cell(row=r, column=6, value=_clean(pick["price"]))
            ws.cell(row=r, column=6).number_format = PRICE_FORMAT
            for i, method in enumerate(methods):
                first = 7 + 5 * i
                top = results[method][query.query_id]
                if position <= len(top):
                    uniq_id = top[position - 1]
                    ws.cell(row=r, column=first, value=uniq_id)
                    ws.cell(row=r, column=first + 1, value=_clean(info.loc[uniq_id, "name"]))
                    ws.cell(row=r, column=first + 2, value=_clean(info.loc[uniq_id, "price"]))
                    ws.cell(row=r, column=first + 3, value=good_of.get((query.query_id, uniq_id), NOT_LABELLED))
                    ws.cell(row=r, column=first + 4, value="yes" if uniq_id in truth_ids else "no")
                ws.cell(row=r, column=first + 2).number_format = PRICE_FORMAT
            r += 1
    last = r - 1
    for i in range(len(methods)):
        first = 7 + 5 * i
        _yes_no_colours(ws, f"{get_column_letter(first + 3)}3:{get_column_letter(first + 4)}{last}")
    ws.freeze_panes = "D3"
    ws.auto_filter.ref = f"A2:{get_column_letter(6 + 5 * len(methods))}{last}"
    widths = {"A": 9, "B": 40, "C": 6, "D": 34, "E": 50, "F": 9}
    for i in range(len(methods)):
        first = 7 + 5 * i
        widths.update({get_column_letter(first): 34, get_column_letter(first + 1): 50, get_column_letter(first + 2): 9,
                       get_column_letter(first + 3): 12, get_column_letter(first + 4): 12})
    _widths(ws, widths)


def summary_sheet(wb: Workbook, queries: pd.DataFrame, methods: list[str]) -> None:
    """The three metrics per test product, as formulas over Details, and the overall numbers below."""
    ws = wb.active
    ws.title = "Summary"
    ws.cell(row=1, column=1, value=f"Similarity evaluation: {len(queries)} test products, top {TOP_K} results each, "
                                   "against a hand-labelled ground truth").font = Font(bold=True)
    _group(ws, 2, 1, 4, "Test product")
    titles = ["Query ID", "Query", "Kind", "Ground-truth size"]
    for i, method in enumerate(methods):
        first = 5 + 3 * i
        _group(ws, 2, first, first + 2, METHODS[method][0])
        titles += ["Good results (of 10)", "Ground truth found", "First good result"]
    _header(ws, 3, titles)

    top, bottom = 4, 3 + len(queries)
    for r, query in enumerate(queries.itertuples(index=False), start=top):
        ws.cell(row=r, column=1, value=query.query_id)
        ws.cell(row=r, column=2, value=query.name)
        ws.cell(row=r, column=3, value=query.kind)
        ws.cell(row=r, column=4, value=f'=COUNTIFS(Details!$A:$A,$A{r},Details!$D:$D,"?*")')
        for i in range(len(methods)):
            first = 5 + 3 * i
            good_col, truth_col = get_column_letter(10 + 5 * i), get_column_letter(11 + 5 * i)
            good_cell = f"{get_column_letter(first)}{r}"
            ws.cell(row=r, column=first, value=f'=COUNTIFS(Details!$A:$A,$A{r},Details!${good_col}:${good_col},"yes")')
            ws.cell(row=r, column=first + 1, value=f'=COUNTIFS(Details!$A:$A,$A{r},Details!${truth_col}:${truth_col},"yes")')
            ws.cell(row=r, column=first + 2, value=(
                f'=IF({good_cell}=0,"none",_xlfn.MINIFS(Details!$C:$C,Details!$A:$A,$A{r},Details!${good_col}:${good_col},"yes"))'
            ))

    rows = [
        ("Average good results (of 10)", lambda c: f"=AVERAGE({c(0)}{top}:{c(0)}{bottom})", "0.0"),
        ("... random test products only", lambda c: f'=AVERAGEIFS({c(0)}{top}:{c(0)}{bottom},$C${top}:$C${bottom},"random")', "0.0"),
        ("... hard test products only", lambda c: f'=AVERAGEIFS({c(0)}{top}:{c(0)}{bottom},$C${top}:$C${bottom},"hard")', "0.0"),
        ("Share of all ground-truth picks found", lambda c: f"=SUM({c(1)}{top}:{c(1)}{bottom})/SUM($D${top}:$D${bottom})", "0%"),
        (f"Average first good result (none = {NONE_RANK})",
         lambda c: f'=(SUM({c(2)}{top}:{c(2)}{bottom})+{NONE_RANK}*COUNTIF({c(2)}{top}:{c(2)}{bottom},"none"))/ROWS({c(2)}{top}:{c(2)}{bottom})', "0.0"),
        ("Test products whose first result is good", lambda c: f"=COUNTIF({c(2)}{top}:{c(2)}{bottom},1)", "0"),
    ]
    for offset, (label, formula, number_format) in enumerate(rows):
        r = bottom + 2 + offset
        ws.cell(row=r, column=2, value=label).font = Font(bold=True)
        for i in range(len(methods)):
            first = 5 + 3 * i
            cell = ws.cell(row=r, column=first, value=formula(lambda k, first=first: get_column_letter(first + k)))
            cell.number_format = number_format
            cell.font = Font(bold=True)

    for i in range(len(methods)):
        first = 5 + 3 * i
        good = f"{get_column_letter(first)}{top}:{get_column_letter(first)}{bottom}"
        ws.conditional_formatting.add(good, ColorScaleRule(start_type="num", start_value=0, start_color="F8696B",
                                                           mid_type="num", mid_value=5, mid_color="FFEB84",
                                                           end_type="num", end_value=10, end_color="63BE7B"))
    ws.freeze_panes = "C4"
    widths = {"A": 9, "B": 62, "C": 8, "D": 11}
    for i in range(len(methods)):
        for k in range(3):
            widths[get_column_letter(5 + 3 * i + k)] = 12
    _widths(ws, widths)
    ws.row_dimensions[3].height = 32


def methods_sheet(wb: Workbook, scores: dict[str, dict[str, float]], timings: dict[str, float]) -> None:
    """One row per method with every ranking metric; each header's note says what it measures."""
    ws = wb.create_sheet("Methods", 0)
    names = [*RANKING_METRICS, "ms per query"]
    _header(ws, 1, ["Method", "Description", *names])
    meanings = {name: meaning for name, (_, meaning) in RANKING_METRICS.items()}
    meanings["ms per query"] = "average time to search for one test product"
    for c, name in enumerate(names, start=3):
        ws.cell(row=1, column=c).comment = Comment(f"{name}: {meanings[name]}. Averaged over the test products.", "evals")
    for r, (method, values) in enumerate(scores.items(), start=2):
        ws.cell(row=r, column=1, value=method)
        ws.cell(row=r, column=2, value=METHODS[method][0])
        for c, name in enumerate(RANKING_METRICS, start=3):
            ws.cell(row=r, column=c, value=round(values[name], 4)).number_format = RANKING_METRICS[name][0]
        ws.cell(row=r, column=3 + len(RANKING_METRICS), value=round(timings[method], 1)).number_format = "0.0"
    ws.freeze_panes = "C2"
    _widths(ws, {"A": 12, "B": 58, **{get_column_letter(c): 12 for c in range(3, 3 + len(names))}})
    ws.row_dimensions[1].height = 30


def apply_font(wb: Workbook) -> None:
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                cell.font = Font(name="Arial", size=10, bold=cell.font.bold, color=cell.font.color)


def build_workbook(queries: pd.DataFrame, labels: pd.DataFrame, results: dict, info: pd.DataFrame, methods: list[str],
                   scores: dict[str, dict[str, float]], timings: dict[str, float]) -> Workbook:
    """Methods (every method's metrics), Summary and Details (the methods shown), Queries and Labels (the data)."""
    wb = Workbook()
    summary_sheet(wb, queries, methods)
    details_sheet(wb, queries, labels, results, info, methods)
    methods_sheet(wb, scores, timings)
    apply_font(wb)
    write_data_sheets(wb, queries, labels)
    wb.active = 0
    wb.calculation.fullCalcOnLoad = True
    return wb


# --- run --------------------------------------------------------------------------------

def unlabelled_rows(results: dict, labels: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Rows for results that have no label yet, to append to the Labels sheet with `good` empty."""
    known = set(zip(labels["query_id"], labels["uniq_id"]))
    missing = {}                                       # (query_id, uniq_id) -> the first method that returned it
    for method, top in results.items():
        for query_id, ids in top.items():
            for uniq_id in ids:
                if (query_id, uniq_id) not in known:
                    missing.setdefault((query_id, uniq_id), method)
    if not missing:
        return pd.DataFrame(columns=labels.columns)
    gender = genders(products)
    rows = []
    for (query_id, uniq_id), method in missing.items():
        row = products.index[products["uniq_id"] == uniq_id][0]
        item = products.loc[row]
        rows.append({
            "query_id": query_id, "position": None, "uniq_id": uniq_id, "name": item["name"],
            "category": item["child_category"], "gender": gender.loc[row], "price": item["price"], "brand": item["brand"],
            "colours": "|".join(item["colours"]) if item["colours"] else None, "good": None, "gt_rank": None,
            "source": method,
        })
    return pd.DataFrame(rows, columns=labels.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alternative", default="none", choices=[*(m for m in METHODS if m != BASELINE), "none"])
    args = parser.parse_args()
    methods = [BASELINE] + ([] if args.alternative == "none" else [args.alternative])

    queries, labels = load_data()
    products = load_products()
    features = load_features()
    info = products.set_index("uniq_id")

    # every registered method is run (the Methods sheet scores them all); Details shows `methods`
    all_results, timings = {}, {}
    for method, (_, find) in METHODS.items():
        try:
            find(features, 0)   # the first call reads the method's files, so it is not timed
        except FileNotFoundError as error:
            if method in methods:
                raise SystemExit(f"{method}: {error}") from error
            print(f"Skipped {method}: {error}")
            continue
        start = time.perf_counter()
        all_results[method] = {
            q.query_id: features.ids[find(features, features.row_of[q.uniq_id])[:TOP_K]].tolist()
            for q in queries.itertuples(index=False)
        }
        timings[method] = (time.perf_counter() - start) / len(queries) * 1000
    results = {method: all_results[method] for method in methods}

    # results nobody has labelled yet go into the Labels sheet with an empty `good` cell
    extra = unlabelled_rows(all_results, labels, products)
    if len(extra):
        labels = pd.concat([labels, extra], ignore_index=True)
    scores = {method: method_scores(top, labels) for method, top in all_results.items()}

    wb = build_workbook(queries, labels, results, info, methods, scores, timings)
    save(wb, WORKBOOK_PATH)

    print(f"Wrote {WORKBOOK_PATH}\n")
    print(pd.DataFrame(scores).T.assign(**{"ms per query": pd.Series(timings)}).round(2).to_string())
    for method in methods:
        table = score(results[method], labels)
        print(f"\n{METHODS[method][0]}")
        for name, value in overall(table).items():
            shown = f"{value:.0%}" if name in ("ground truth found", "first result good") else f"{value:.1f}"
            print(f"  {name:<22} {shown}")
        by_kind = table.join(queries.set_index("query_id")["kind"]).groupby("kind")["good_results"].mean()
        print(f"  {'good results by kind':<22} " + ", ".join(f"{k} {v:.1f}" for k, v in by_kind.items()))
    waiting = int((~labels["good"].isin(["yes", "no"])).sum())
    if waiting:
        print(f"\nWARNING: {waiting} results have no label yet. They show as '{NOT_LABELLED}' in Details and count as")
        print("not good, so the scores above are incomplete. On the Labels sheet, fill their empty `good` cells")
        print("with yes / no, close the workbook, and run again.")


if __name__ == "__main__":
    main()
