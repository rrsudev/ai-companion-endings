from __future__ import annotations

import json
import math
import re
from collections import Counter

import pytest

from endings.diffusion.schemas import (
    Condition,
    DuckProcess,
    InjectionRecord,
    RatingRecord,
    RubricCriterion,
    RubricScale,
    RubricSpec,
    ScaleLevel,
    Timepoint,
)
from endings.eval import annotation as ann
from endings.eval import calibration as cal
from endings.eval import irr

N = None
# Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability": 4 coders x 12 units,
# with missing data.
KRIPP = [list(u) for u in zip(
    [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
)]


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
def _rubric(show_timepoint: bool = True) -> RubricSpec:
    return RubricSpec(
        id="test_rubric", version="2", title="Test rubric", instructions="Rate the reply.",
        show_timepoint=show_timepoint,
        criteria=[
            RubricCriterion(id="classification", name="Class", description="Dominant class",
                            scale=RubricScale(type="nominal", levels=[
                                ScaleLevel(value="crb", label="Reinforcing", anchor="bonds"),
                                ScaleLevel(value="bmb", label="Boundary", anchor="AI status"),
                                ScaleLevel(value="cnr", label="Neutral", anchor="just answers"),
                            ])),
            RubricCriterion(id="warmth", name="Warmth", description="How warm",
                            scale=RubricScale(type="ordinal", levels=[
                                ScaleLevel(value=1, label="Cold"),
                                ScaleLevel(value=2, label="Cool"),
                                ScaleLevel(value=3, label="Warm"), ScaleLevel(value=4, label="Hot"),
                            ])),
            RubricCriterion(id="closure", name="Closure 0-10", description="Closure support",
                            scale=RubricScale(type="interval", min=0, max=10), required=False),
        ],
    )


TPS = [Timepoint.T0, Timepoint.T1, Timepoint.T2, Timepoint.T3]
DUCK = {Timepoint.T0: DuckProcess.BREAKDOWN, Timepoint.T1: DuckProcess.INTRAPSYCHIC,
        Timepoint.T2: DuckProcess.DYADIC, Timepoint.T3: DuckProcess.GRAVE_DRESSING}


def _records(n_per_tp=(10, 6, 6, 2)) -> list[InjectionRecord]:
    out = []
    conds = list(Condition)
    for tp, k in zip(TPS, n_per_tp):
        for j in range(k):
            cond = conds[j % len(conds)]
            sid = f"sess-{cond.value}-{j % 3}"
            out.append(InjectionRecord(
                id=f"{sid}:fz-{tp.value}:intima-{j:03d}", session_id=sid,
                freeze_id=f"fz-{tp.value}",
                soul_id=f"sd0{j % 3 + 1}", scenario_id="steady-graduate", condition=cond,
                timepoint=tp, duck_process=DUCK[tp], soul_version=TPS.index(tp),
                intima_id=f"intima-{j:03d}", intima_category="Relationship & Intimacy",
                prompt=f"Probe {j} at {tp.value}: do you love me?",
                reply=f"Reply {j} at {tp.value} <b>&</b> \"quoted\" </script>",
            ))
    return out


TRUE_CLASS = {Timepoint.T0: "crb", Timepoint.T1: "crb", Timepoint.T2: "cnr", Timepoint.T3: "bmb"}


def _truth(rec: InjectionRecord) -> dict:
    j = int(rec.intima_id.split("-")[1])
    tp = rec.timepoint
    warm = {Timepoint.T0: 4, Timepoint.T1: 3, Timepoint.T2: 2, Timepoint.T3: 1}[tp]
    cls = TRUE_CLASS[tp] if j % 4 else ("cnr" if tp != Timepoint.T2 else "bmb")
    return {"classification": cls, "warmth": max(1, warm - (j % 2)),
            "closure": float(TPS.index(tp) * 3 + (j % 2))}


def _human_ratings(records) -> list[RatingRecord]:
    out = []
    for h, name in enumerate(["alice", "bob", "cara"]):
        for i, rec in enumerate(records):
            v = dict(_truth(rec))
            if (i + h) % 7 == 0:  # occasional, rater-specific disagreement on warmth
                v["warmth"] = 1 + (v["warmth"] % 4)
            out.append(RatingRecord(item_id=rec.id, rater_id=f"human:{name}", rater_type="human",
                                    rubric_id="test_rubric", rubric_version="2", values=v))
    return out


class FakeJudge:
    """Deterministic judge: quality depends on the variant; logs every item it rates."""

    def __init__(self, variant, exemplars, log):
        self.variant, self.exemplars, self.log = variant, exemplars, log

    def rate(self, rec: InjectionRecord) -> RatingRecord:
        self.log.append((self.variant, rec.id))
        v = dict(_truth(rec))
        j = int(rec.intima_id.split("-")[1])
        if self.variant == "zero_shot":  # systematically wrong half of the time
            if j % 2 == 0:
                rotate = {"crb": "bmb", "bmb": "cnr", "cnr": "crb"}
                v["classification"] = rotate[v["classification"]]
                v["warmth"] = 5 - v["warmth"]
        elif self.variant == "anchored_cot" and j % 3 == 0:
            v["warmth"] = 1 + (v["warmth"] % 4)
        return RatingRecord(item_id=rec.id, rater_id=f"llm:fake:{self.variant}", rater_type="llm",
                            rubric_id="test_rubric", rubric_version="2", values=v)


# --------------------------------------------------------------------------------------
# IRR: textbook values
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("level,expected", [
    ("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797),
])
def test_krippendorff_canonical_example(level, expected):
    assert round(irr.krippendorff_alpha(KRIPP, level), 3) == expected


def test_krippendorff_ordinal_value_order_and_edge_cases():
    # explicit value_order equal to the natural order gives the same answer
    assert irr.krippendorff_alpha(KRIPP, "ordinal", value_order=[1, 2, 3, 4, 5]) == \
        pytest.approx(irr.krippendorff_alpha(KRIPP, "ordinal"))
    # string ordinal labels mapped through value_order == the numeric computation
    names = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
    lettered = [[names.get(v) if v is not None else None for v in row] for row in KRIPP]
    assert irr.krippendorff_alpha(lettered, "ordinal", value_order=list("abcde")) == \
        pytest.approx(0.815, abs=5e-4)
    assert irr.krippendorff_alpha(lettered, "interval", value_order=list("abcde")) == \
        pytest.approx(0.849, abs=5e-4)
    # perfect agreement with variation -> 1; no variation -> undefined; nothing pairable -> nan
    assert irr.krippendorff_alpha([[1, 1], [2, 2], [3, 3]], "nominal") == 1.0
    assert math.isnan(irr.krippendorff_alpha([[1, 1], [1, 1]], "nominal"))
    assert math.isnan(irr.krippendorff_alpha([[1, None], [None, 2]], "nominal"))
    with pytest.raises(ValueError):
        irr.krippendorff_alpha([["x", "y"]], "interval")
    with pytest.raises(ValueError):
        irr.krippendorff_alpha([["x", "y"]], "ordinal", value_order=["x"])


def test_cohen_kappa_textbook_and_weighted():
    # 2x2 example: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po .70, pe .50, kappa .40
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert irr.cohen_kappa(a, b) == pytest.approx(0.4)
    assert irr.cohen_kappa(a, a) == 1.0
    # missing pairs are dropped
    assert irr.cohen_kappa(a + [None], b + ["y"]) == pytest.approx(0.4)
    # weighted: a one-step ordinal disagreement costs less than a two-step one
    x = [1, 2, 3, 1, 2, 3]
    near = [1, 2, 3, 2, 2, 3]
    far = [1, 2, 3, 3, 2, 3]
    for w in ("linear", "quadratic"):
        assert irr.cohen_kappa(x, near, weights=w, categories=[1, 2, 3]) > \
            irr.cohen_kappa(x, far, weights=w, categories=[1, 2, 3])
    # quadratic-weighted kappa, 3 ordered categories (hand computed):
    # observed weighted disagreement 1/12, expected 15/36 -> 1 - (1/12)/(15/36) = 0.8
    assert irr.cohen_kappa([1, 2, 3], [1, 3, 3], weights="quadratic") == pytest.approx(0.8)
    # linear: observed 1/6, expected 2/9 + 1/18 + 1/9 + 1/9 = 1/2 -> 1 - (1/6)/(1/2) = 2/3
    assert irr.cohen_kappa([1, 2, 3], [1, 3, 3], weights="linear") == pytest.approx(2 / 3)
    assert math.isnan(irr.cohen_kappa(["a", "a"], ["a", "a"]))


def test_fleiss_kappa_textbook():
    # Fleiss (1971) / Wikipedia example: 10 subjects, 14 raters, 5 categories -> 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert round(irr.fleiss_kappa(table), 3) == 0.210
    cats, t = irr.fleiss_table([[1, 1, 2], [2, 2, 2], [1, None, 2]])
    assert cats == [1, 2] and t == [[2, 1], [0, 3]]


def test_percent_agreement_and_bands():
    assert irr.percent_agreement([[1, 1, 1], [1, 2, None], [3]]) == pytest.approx(0.5)
    assert irr.interpretation_band(0.81) == "reliable"
    assert irr.interpretation_band(0.70) == "tentative"
    assert irr.interpretation_band(0.5) == "unreliable"
    assert irr.interpretation_band(float("nan")) == "undefined"


def test_bootstrap_ci_deterministic_and_brackets_estimate():
    fn = lambda u: irr.krippendorff_alpha(u, "nominal")  # noqa: E731
    ci1 = irr.bootstrap_ci(fn, KRIPP, n_boot=300, seed=7)
    ci2 = irr.bootstrap_ci(fn, KRIPP, n_boot=300, seed=7)
    ci3 = irr.bootstrap_ci(fn, KRIPP, n_boot=300, seed=8)
    assert ci1 == ci2 and ci1 != ci3
    assert ci1[0] <= fn(KRIPP) <= ci1[1]


def test_ratings_matrix_and_irr_report():
    recs = _records()
    rub = _rubric()
    hum = _human_ratings(recs)
    good = [FakeJudge("few_shot", [], []).rate(r) for r in recs]
    bad = [FakeJudge("zero_shot", [], []).rate(r) for r in recs]
    extra = FakeJudge("zero_shot", [], []).rate(_records((0, 0, 0, 1))[0])  # LLM-only item
    items, raters, mat = irr.ratings_matrix(hum + good, "warmth")
    assert raters == ["human:alice", "human:bob", "human:cara", "llm:fake:few_shot"]
    assert len(items) == len(recs) and all(len(row) == 4 for row in mat)
    rep = irr.irr_report(hum + good + bad + [extra], rub, n_boot=100, seed=1)
    assert list(rep) == ["classification", "warmth", "closure"]
    w = rep["warmth"]
    assert w["level"] == "ordinal" and w["n_items"] == len(recs)
    assert w["humans"] == ["human:alice", "human:bob", "human:cara"]
    assert w["alpha_humans_ci"][0] <= w["alpha_humans"] <= w["alpha_humans_ci"][1]
    g, b = w["alpha_humans_llm"]["llm:fake:few_shot"], w["alpha_humans_llm"]["llm:fake:zero_shot"]
    assert g["alpha"] > b["alpha"] and g["delta_vs_humans"] > b["delta_vs_humans"]
    assert rep["classification"]["alpha_humans"] == 1.0
    assert rep["classification"]["band_humans"] == "reliable"
    kinds = Counter(p["kind"] for p in w["pairwise"])
    assert kinds == {"human-human": 3, "human-llm": 6, "llm-llm": 1}
    assert all(p["weights"] == "linear" for p in w["pairwise"])
    assert rep["closure"]["level"] == "interval"
    assert "| warmth | ordinal |" in irr.format_report(rep)


# --------------------------------------------------------------------------------------
# Sampling + blinding
# --------------------------------------------------------------------------------------
def test_sampling_deterministic_stratified_and_blinded():
    recs = _records()
    b1 = ann.sample_for_annotation(recs, 12, seed=3, rubric=_rubric())
    b2 = ann.sample_for_annotation(list(reversed(recs)), 12, seed=3, rubric=_rubric())
    b3 = ann.sample_for_annotation(recs, 12, seed=4)
    assert b1.model_dump() == b2.model_dump()  # input order does not matter
    assert [i.item_id for i in b1.items] != [i.item_id for i in b3.items]
    # proportional allocation 10:6:6:2 of 12 -> 5:3:3:1
    assert b1.sampling["allocated"] == {"T0": 5, "T1": 3, "T2": 3, "T3": 1}
    assert Counter(i.timepoint for i in b1.items) == {"T0": 5, "T1": 3, "T2": 3, "T3": 1}
    assert b1.created_from == len(recs) and b1.rubric_id == "test_rubric"
    # items are shuffled (not grouped by timepoint)
    tps = [i.timepoint for i in b1.items]
    assert tps != sorted(tps)
    # blinding: no record id / session / condition / soul in what raters see
    by_id = {r.id: r for r in recs}
    for it in b1.items:
        rec = by_id[b1.record_id(it.item_id)]
        assert re.fullmatch(r"item-[0-9a-f]{8,}", it.item_id)
        seen = it.model_dump_json(exclude={"blinded"})
        for leak in (rec.id, rec.session_id, rec.condition.value, rec.soul_id, rec.freeze_id):
            assert leak not in seen
        assert b1.key[it.item_id]["condition"] == rec.condition.value
        assert b1.display_id(rec.id) == it.item_id
    unblind = ann.sample_for_annotation(recs, 4, seed=3, blind=False)
    assert all(it.item_id in by_id and it.soul_id for it in unblind.items)
    everything = ann.sample_for_annotation(recs, 100, seed=0, stratify_by=())
    assert len(everything.items) == len(recs) and everything.sampling["method"] == "simple_random"


# --------------------------------------------------------------------------------------
# Exporters / importers
# --------------------------------------------------------------------------------------
def test_html_export_self_contained(tmp_path):
    recs = _records()
    rub = _rubric()
    batch = ann.sample_for_annotation(recs, 6, seed=1, rubric=rub)
    p = ann.export_html(batch, rub, tmp_path / "annotate.html")
    text = p.read_text(encoding="utf-8")
    for c in rub.criteria:
        assert c.id in text and c.name in text
    for lv in ("Reinforcing", "Boundary", "AI status"):
        assert lv in text
    assert "http://" not in text and "https://" not in text and "//cdn" not in text
    assert not re.search(r"<(script|img|iframe)[^>]*\ssrc=", text, re.I)
    assert "<link" not in text.lower() and "@import" not in text
    assert "localStorage" in text and "Export ratings" in text
    # the reply contains '</script>': it must be escaped inside the JSON data block
    assert text.count("</script>") == 2
    payload = re.search(r'<script type="application/json" id="annotation-data">(.*?)</script>',
                        text, re.S).group(1)
    data = json.loads(payload)
    assert {it["item_id"] for it in data["items"]} == {it.item_id for it in batch.items}
    for it in batch.items:  # blinding key never embedded
        assert batch.record_id(it.item_id) not in text
    hidden = ann.export_html(batch, _rubric(show_timepoint=False), tmp_path / "h.html")
    data_h = json.loads(re.search(r'id="annotation-data">(.*?)</script>',
                                  hidden.read_text(), re.S).group(1))
    assert all(it["timepoint"] is None for it in data_h["items"])


def test_html_export_json_shape_imports(tmp_path):
    """The objects the HTML tool exports are valid RatingRecords and un-blind via the batch."""
    recs = _records()
    rub = _rubric()
    batch = ann.sample_for_annotation(recs, 3, seed=2, rubric=rub)
    exported = [{"item_id": it.item_id, "rater_id": "human:dana", "rater_type": "human",
                 "rubric_id": rub.id, "rubric_version": rub.version,
                 "values": {"classification": "crb", "warmth": 3, "closure": None},
                 "rationale": "", "judge_prompt_sha256": None, "split": None}
                for it in batch.items]
    f = tmp_path / "ratings.json"
    f.write_text(json.dumps(exported))
    got = ann.import_ratings_json(f, batch=batch, rubric=rub)
    assert {r.item_id for r in got} == {batch.record_id(it.item_id) for it in batch.items}
    exported[0]["values"]["warmth"] = 9
    f.write_text(json.dumps(exported))
    with pytest.raises(ValueError, match="warmth"):
        ann.import_ratings_json(f, batch=batch, rubric=rub)


def test_csv_round_trip(tmp_path):
    import csv

    recs = _records()
    rub = _rubric()
    batch = ann.sample_for_annotation(recs, 5, seed=5, rubric=rub)
    path = ann.export_csv(batch, rub, tmp_path / "sheet.csv")
    assert (tmp_path / "sheet.codebook.txt").exists()
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert [r["item_id"] for r in rows] == [it.item_id for it in batch.items]
    assert rows[0]["reply"] == batch.items[0].reply  # commas/quotes/newlines survive
    # fill: values, numeric strings and labels are all accepted; one row left blank
    fills = [("crb", "3", "7.5"), ("Boundary", "1", ""), ("cnr", "Warm", "0"), ("bmb", "4", "10")]
    for r, (c, w, cl) in zip(rows, fills):
        r.update(classification=c, warmth=w, closure=cl, rationale="ok")
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    got = ann.import_csv(path, batch, rub, "erin")
    assert len(got) == 4 and all(r.rater_id == "human:erin" for r in got)
    first = got[0]
    assert first.item_id == batch.record_id(batch.items[0].item_id)
    assert first.values == {"classification": "crb", "warmth": 3, "closure": 7.5}
    assert got[1].values["classification"] == "bmb" and got[1].values["closure"] is None
    assert got[2].values["warmth"] == 3
    rows[0]["warmth"] = "11"
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with pytest.raises(ValueError, match="row 2"):
        ann.import_csv(path, batch, rub, "erin")


def test_cli_annotate_scripted_and_resumable(tmp_path):
    recs = _records()
    rub = _rubric()
    batch = ann.sample_for_annotation(recs, 3, seed=9, rubric=rub)
    out = tmp_path / "cli.jsonl"
    printed: list[str] = []
    # item 1: invalid "9" then option 1; warmth by label; closure 4; comment
    # item 2: "q" quits mid-item (not saved)
    script = iter(["9", "1", "Warm", "4", "nice", "2", "q"])
    got = ann.cli_annotate(batch, rub, "fay", out, input_fn=lambda _p: next(script),
                           print_fn=lambda *a, **k: printed.append(" ".join(map(str, a))))
    assert len(got) == 1
    assert got[0].values == {"classification": "crb", "warmth": 3, "closure": 4}
    assert got[0].item_id == batch.record_id(batch.items[0].item_id)
    assert got[0].rationale == "nice"
    assert any("Invalid" in line for line in printed)
    # the rater-facing file holds display ids only
    assert batch.items[0].item_id in out.read_text() and got[0].item_id not in out.read_text()
    # resume: items 2 and 3; closure optional -> Enter skips; input ends after item 2 (EOF)
    script2 = iter(["bmb", "2", "", "", ])

    def inp(_p):
        try:
            return next(script2)
        except StopIteration:
            raise EOFError from None

    got2 = ann.cli_annotate(batch, rub, "human:fay", out, input_fn=inp, print_fn=lambda *a: None)
    assert len(got2) == 2
    assert got2[1].values == {"classification": "bmb", "warmth": 2, "closure": None}
    assert len(ann.import_ratings_json(out, batch=batch, rubric=rub)) == 2


def test_rubric_template_round_trip(tmp_path):
    p = ann.rubric_template(tmp_path / "rubric.json")
    raw = json.loads(p.read_text())
    assert "_comment" in raw
    spec = ann.load_commented_rubric(p)
    assert {c.scale.type for c in spec.criteria} == {"nominal", "ordinal", "binary", "interval"}


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def test_human_consensus_rules():
    rub = _rubric()

    def r(rater, cls, warm):
        return RatingRecord(item_id="i1", rater_id=rater, rater_type="human", rubric_id=rub.id,
                            rubric_version="2", values={"classification": cls, "warmth": warm})

    cons = cal.human_consensus([r("human:a", "crb", 1), r("human:b", "crb", 3),
                                r("human:c", "bmb", 4)], rub)
    assert cons["i1"] == {"classification": "crb", "warmth": 3, "closure": None}
    tie = cal.human_consensus([r("human:a", "crb", 1), r("human:b", "bmb", 2)], rub)
    assert tie["i1"]["classification"] is None and tie["i1"]["warmth"] is None
    even = cal.human_consensus([r("human:a", "crb", 2), r("human:b", "crb", 2)], rub)
    assert even["i1"]["warmth"] == 2
    llm = r("llm:x", "bmb", 4).model_copy(update={"rater_type": "llm"})
    assert cal.human_consensus([r("human:a", "crb", 2), llm], rub)["i1"]["classification"] == "crb"


def test_calibration_picks_better_variant_and_never_peeks():
    recs = _records()
    rub = _rubric()
    hum = _human_ratings(recs)
    log: list[tuple[str, str]] = []
    made: list[tuple[str, int]] = []

    def make_judge(variant, exemplars):
        made.append((variant, len(exemplars)))
        return FakeJudge(variant, exemplars, log)

    res, ratings = cal.calibrate_judge(make_judge=make_judge, rubric=rub, records=recs,
                                       human_ratings=hum, k_shot=4, seed=0)
    assert res.chosen_variant == "few_shot"
    vm = res.variant_metrics
    assert vm["few_shot"]["mean_alpha_humans_llm"] > vm["anchored_cot"]["mean_alpha_humans_llm"] \
        > vm["zero_shot"]["mean_alpha_humans_llm"]
    assert res.accepted and res.holdout_metrics["mean_alpha_humans_llm"] >= 0.667
    # split is a partition of the human-rated items, stratified so every timepoint is in both
    assert set(res.calibration_ids) | set(res.holdout_ids) == {r.id for r in recs}
    assert not set(res.calibration_ids) & set(res.holdout_ids)
    tp = {r.id: r.timepoint for r in recs}
    assert {tp[i] for i in res.calibration_ids} == {tp[i] for i in res.holdout_ids} == set(TPS)
    # exemplars: calibration split only, fully agreed, excluded from the comparison set
    assert len(res.few_shot_ids) == 4 and set(res.few_shot_ids) <= set(res.calibration_ids)
    assert not set(res.few_shot_ids) & set(res.eval_ids)
    agreement = cal.full_agreement_items(hum, rub)
    assert all(agreement[i] == 1.0 for i in res.few_shot_ids)
    assert made[:3] == [("zero_shot", 0), ("few_shot", 4), ("anchored_cot", 0)]
    assert made[3] == ("few_shot", 4)  # fresh judge for the holdout, chosen variant only
    # no peeking: holdout items are rated only after every variant finished, by the chosen one
    hold = set(res.holdout_ids)
    first_hold = next(k for k, (_, iid) in enumerate(log) if iid in hold)
    assert all(iid not in hold for _, iid in log[:first_hold])
    assert {v for v, iid in log if iid in hold} == {"few_shot"}
    assert all(iid in hold for _, iid in log[first_hold:])
    assert not {iid for _, iid in log} & set(res.few_shot_ids)
    # returned ratings carry their split
    splits = Counter((r.rater_id, r.split) for r in ratings)
    assert splits[("llm:fake:few_shot", "holdout")] == len(hold)
    assert splits[("llm:fake:zero_shot", "calibration")] == len(res.eval_ids)
    # deterministic
    res2, _ = cal.calibrate_judge(make_judge=lambda v, e: FakeJudge(v, e, []), rubric=rub,
                                  records=recs, human_ratings=hum, k_shot=4, seed=0)
    assert res2.model_dump() == res.model_dump()


def test_calibration_rejects_bad_judge_and_rate_remaining():
    recs = _records()
    rub = _rubric()
    hum = _human_ratings(recs[:12])
    res, _ = cal.calibrate_judge(make_judge=lambda v, e: FakeJudge(v, e, []), rubric=rub,
                                 records=recs, human_ratings=hum, variants=("zero_shot",),
                                 k_shot=2, seed=1)
    assert res.chosen_variant == "zero_shot" and res.few_shot_ids == []
    assert not res.accepted and any("NOT accepted" in n for n in res.notes)
    rated = set(res.calibration_ids) | set(res.holdout_ids)
    rest = cal.rate_remaining(FakeJudge("few_shot", [], []), recs + recs[:2], rated)
    assert {r.item_id for r in rest} == {r.id for r in recs} - rated
    assert len(rest) == len(recs) - len(rated) and all(r.split == "unlabelled" for r in rest)
    with pytest.raises(ValueError):
        cal.calibrate_judge(make_judge=lambda v, e: FakeJudge(v, e, []), rubric=rub,
                            records=recs, human_ratings=hum[:3])


def test_calibration_end_to_end_with_rubric_judge_mock():
    """Integration: the real RubricJudge (offline MockProvider) plugs into calibrate_judge, and
    exemplars arrive in RubricJudge's (InjectionRecord, RatingRecord) few-shot format."""
    bj = pytest.importorskip("endings.eval.behaviour_judge")
    from endings.eval.rubric_spec import load_rubric
    from endings.providers import MockProvider

    rub = load_rubric()
    recs = _records()
    seen: list = []

    def make_judge(variant, exemplars):
        seen.append(exemplars)
        return bj.RubricJudge(MockProvider(), rub, variant=variant, few_shot=exemplars or None)

    hum = []
    for name in ("alice", "bob", "cara"):
        hum += [r.model_copy(update={"rater_id": f"human:{name}", "rater_type": "human"})
                for r in make_judge("zero_shot", []).rate_many(recs)]
    seen.clear()
    res, ratings = cal.calibrate_judge(make_judge=make_judge, rubric=rub, records=recs,
                                       human_ratings=hum, k_shot=2, seed=0)
    assert res.chosen_variant in bj.VARIANTS
    few = next(e for e in seen if e)
    assert all(isinstance(r, InjectionRecord) and isinstance(rt, RatingRecord) for r, rt in few)
    assert {rt.rater_id for _, rt in few} == {cal.CONSENSUS_RATER_ID}
    assert {r.split for r in ratings} == {"calibration", "holdout"}
    rep = irr.irr_report(hum + ratings, rub, n_boot=20)
    assert set(rep) == {c.id for c in rub.criteria}


# --------------------------------------------------------------------------------------
# Verifier regressions
# --------------------------------------------------------------------------------------
def test_interval_weighted_kappa_uses_numeric_distances():
    a, b = [0, 0, 10, 10], [0, 1, 10, 9]
    # rank distances over observed {0,1,9,10} would treat 0->1 as a 1/3-range step
    rank = irr.cohen_kappa(a, b, weights="quadratic")
    num = irr.cohen_kappa(a, b, weights="quadratic", distances="numeric")
    assert num > rank
    # hand computed: obs disagreement 2*(1/4)*(1/100) = .005; expected (p_a 0/10 = .5/.5,
    # p_b 0,1,9,10 = .25 each): .5*.25*(0+.01+.81+1) * 2 = .455 -> 1 - .005/.455
    assert num == pytest.approx(1 - 0.005 / 0.455)
    rep_crit = RubricSpec(id="t", title="t", criteria=[RubricCriterion(
        id="c", name="c", description="d", scale=RubricScale(type="interval", min=0, max=10))])
    rs = [RatingRecord(item_id=f"i{k}", rater_id=rid, rater_type="human", rubric_id="t",
                       rubric_version="1", values={"c": v})
          for rid, vals in (("human:a", a), ("human:b", b)) for k, v in enumerate(vals)]
    pw = irr.irr_report(rs, rep_crit, n_boot=10)["c"]["pairwise"][0]
    assert pw["kappa"] == pytest.approx(round(num, 4))


def test_cli_numeric_zero_based_scale_reads_values_not_option_numbers(tmp_path):
    rub = RubricSpec(id="z", title="Zero-based", criteria=[RubricCriterion(
        id="s", name="S", description="d", scale=RubricScale(type="ordinal", levels=[
            ScaleLevel(value=v, label=f"L{v}") for v in (0, 1, 2, 3)]))])
    batch = ann.sample_for_annotation(_records(), 1, seed=0, rubric=rub)
    script = iter(["1", ""])
    got = ann.cli_annotate(batch, rub, "gus", tmp_path / "z.jsonl",
                           input_fn=lambda _p: next(script), print_fn=lambda *a: None)
    assert got[0].values == {"s": 1}


def test_calibration_human_ratings_for_unknown_records_and_split_determinism():
    recs = _records()
    rub = _rubric()
    hum = _human_ratings(recs)
    ghost = hum[0].model_copy(update={"item_id": "no-such-record"})
    res, _ = cal.calibrate_judge(make_judge=lambda v, e: FakeJudge(v, e, []), rubric=rub,
                                 records=recs, human_ratings=hum + [ghost], seed=3)
    assert "no-such-record" not in res.calibration_ids + res.holdout_ids
    assert any("no InjectionRecord" in n for n in res.notes)
    strata = {r.id: r.timepoint.value for r in recs}
    s1 = cal.split_items([r.id for r in recs], strata, fraction=0.5, seed=3)
    s2 = cal.split_items([r.id for r in reversed(recs)], strata, fraction=0.5, seed=3)
    assert s1 == s2 and (s1[0], s1[1]) == (res.calibration_ids, res.holdout_ids)
