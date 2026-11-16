import re

import pandas as pd
import pytest

from evals.run_eval import (
    METHODS,
    build_workbook,
    first_good,
    good_results,
    ground_truth_found,
    method_scores,
    overall,
    ranking_metrics,
    score,
)
from evals.store import WORKBOOK_PATH, load_data, save_data


def toy_labels():
    rows = [
        ("Q01", "a", "yes", 1), ("Q01", "b", "yes", 2), ("Q01", "c", "no", None), ("Q01", "d", "yes", None),
        ("Q02", "e", "no", None), ("Q02", "f", "yes", 1),
    ]
    labels = pd.DataFrame(rows, columns=["query_id", "uniq_id", "good", "gt_rank"])
    return labels.assign(position=range(1, 7), name=list("ABCDEF"), category=None, gender=None,
                         price=[100.0, None, 300.0, 400.0, 500.0, 600.0], brand=None, colours=None, source="shortlist")


def toy_queries():
    return pd.DataFrame({"query_id": ["Q01", "Q02"], "kind": ["random", "hard"], "why": ["x", "y"],
                         "uniq_id": ["q1", "q2"], "name": ["Query one", "Query two"], "category": None,
                         "gender": None, "price": [1.0, None]})


def test_metrics():
    assert good_results(["a", "c", "d"], {"a", "d"}) == 2
    assert ground_truth_found(["d", "a"], ["a", "b"]) == 1
    assert first_good(["c", "x", "a"], {"a"}) == 3
    assert first_good(["c"], {"a"}) is None


def test_score_counts_unlabelled_separately():
    table = score({"Q01": ["c", "a", "z"], "Q02": ["e"]}, toy_labels())
    assert table.loc["Q01"].to_dict() == {
        "good_results": 1, "ground_truth_found": 1, "ground_truth_size": 2, "first_good": 2, "not_labelled": 1,
    }
    assert pd.isna(table.loc["Q02", "first_good"])


def test_overall_counts_no_good_result_as_rank_11():
    table = score({"Q01": ["a"], "Q02": ["e"]}, toy_labels())
    numbers = overall(table)
    assert numbers["first good result"] == (1 + 11) / 2
    assert numbers["ground truth found"] == 1 / 3     # 1 of the 3 picks across both queries
    assert numbers["first result good"] == 0.5


def test_ranking_metrics_by_hand():
    # picks a, b, c, q; the method finds a at rank 2, b at 4, c at 6, and never q
    top = ["x", "a", "y", "b", "z", "c", "d", "e", "f", "g"]
    numbers = ranking_metrics(top, ["a", "b", "c", "q"], good={"a", "x"})
    assert numbers == pytest.approx({"Recall@5": 2 / 4, "Recall@10": 3 / 4, "MRR@10": 1 / 2, "Good@10": 2 / 10})
    miss = ranking_metrics(["x", "y"], ["a"], good=set())
    assert (miss["Recall@5"], miss["Recall@10"], miss["MRR@10"]) == (0, 0, 0)


def test_workbook_layout():
    labels = toy_labels()
    info = pd.concat([labels.set_index("uniq_id")[["name", "price"]],
                      pd.DataFrame({"name": ["Never labelled"], "price": [1.0]}, index=["zz"])])
    results = {"tfidf": {"Q01": ["a", "c"], "Q02": ["f", "zz"]}}
    scores = {"tfidf": method_scores(results["tfidf"], labels)}
    wb = build_workbook(toy_queries(), labels, results, info, ["tfidf"], scores, {"tfidf": 21.5})
    assert wb.sheetnames == ["Methods", "Summary", "Details", "Queries", "Labels"]
    methods = wb["Methods"]
    assert (methods["A2"].value, methods["C1"].value) == ("tfidf", "Recall@5")
    assert methods["C2"].value == pytest.approx((1 / 2 + 1 / 1) / 2)     # Recall@5: Q01 found 1 of 2, Q02 1 of 1
    assert methods["C1"].comment is not None                             # each metric explains itself on hover

    summary_formulas = [c.value for row in wb["Summary"].iter_rows() for c in row
                        if isinstance(c.value, str) and c.value.startswith("=")]
    assert summary_formulas and wb["Summary"]["E4"].value.startswith("=COUNTIFS(Details!$A:$A,$A4,Details!$J:$J")
    referenced = {name for f in summary_formulas for name in re.findall(r"([A-Za-z_][\w ]*)!", f)}
    assert referenced == {"Details"}                                             # Summary reads Details only

    details = wb["Details"]
    assert details.max_column == 11                                              # one method, no second column
    assert [details[f"{c}3"].value for c in "DGJK"] == ["a", "a", "yes", "yes"]  # rank 1: in ground truth, good
    assert [details[f"{c}4"].value for c in "DGJK"] == ["b", "c", "no", "no"]
    assert details["J14"].value == "not labelled"                                # Q02 rank 2 was never labelled
    assert wb["Labels"].max_row == len(labels) + 1


def test_data_sheets_round_trip(tmp_path):
    path = tmp_path / "eval.xlsx"
    records = toy_labels().to_dict("records")
    records.append({**records[0], "uniq_id": "new", "good": None, "gt_rank": None, "source": "minilm"})  # awaiting a label
    labels = pd.DataFrame(records)
    save_data(toy_queries(), labels, path)
    queries_back, labels_back = load_data(path)
    assert queries_back["uniq_id"].tolist() == ["q1", "q2"]
    assert labels_back["uniq_id"].tolist() == labels["uniq_id"].tolist()
    assert labels_back["good"].tolist()[:6] == labels["good"].tolist()[:6] and pd.isna(labels_back["good"].iloc[6])
    assert labels_back["gt_rank"].tolist()[:2] == [1, 2]


@pytest.mark.skipif(not WORKBOOK_PATH.exists(), reason="no workbook")
def test_labels_are_consistent():
    try:
        _, labels = load_data()
    except PermissionError:
        pytest.skip("the workbook is open in Excel")
    picks = labels.dropna(subset=["gt_rank"])
    assert set(labels["good"].dropna()) <= {"yes", "no"}
    assert set(labels["source"]) <= {"shortlist", "sweep", *METHODS}       # round one, the sweep, or a method's result
    assert (picks["good"] == "yes").all()                                  # every pick is a good one
    assert not picks.duplicated(["query_id", "gt_rank"]).any()
    assert picks.groupby("query_id").size().max() <= 10
    assert not labels.duplicated(["query_id", "uniq_id"]).any()
