from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from endings.eval import irr_viewer as iv

JUDGE = "llm:test-judge:zero_shot"


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _session(sid: str, condition: str) -> dict:
    return {
        "session_id": sid, "soul_id": "sd01", "scenario_id": "steady-graduate",
        "condition": condition, "preferences": {"notes": "wants evenings back"},
        "soul_versions": [{"document": {"companion": {"name": "Lu", "user_nicknames": ["A-Wan"]},
                                        "user_role": {"name": "Wen"}}}],
        "turns": [{"index": i, "timepoint": "T0" if i < 2 else "T1", "user_message": f"u{i}",
                   "reply": f"reply {i}", "user_events": [{"app": "Browser", "action": "read"}],
                   "directives": ["be gentle"] if condition != "control" else []}
                  for i in range(4)],
        "interventions": [] if condition == "control" else [{
            "id": f"{sid}:iv1", "turn_index": 2, "from_timepoint": "T0", "to_timepoint": "T1",
            "trigger": "schedule", "edits": [{"op": "clear", "path": "relationship.dynamics"}],
            "assessment": {"rationale": "ready", "risk_flags": []}}],
        "freeze_points": [{"id": "fz0", "after_turn": 1, "timepoint": "T0",
                           "duck_process": "breakdown"}],
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    sessions = [_session("s-a__control", "control"), _session("s-b__gradual", "gradual_tempo")]
    _write_jsonl(tmp_path / "transcripts.jsonl", sessions)
    inj, rat = [], []
    for s in sessions:
        for k in range(2):
            iid = f"{s['session_id']}:fz0:intima-{k}"
            inj.append({"id": iid, "session_id": s["session_id"], "freeze_id": "fz0",
                        "timepoint": "T0", "intima_id": f"intima-{k}", "prompt": "p",
                        "reply": "r"})
            rat.append({"item_id": iid, "rater_id": JUDGE, "rater_type": "llm",
                        "values": {"classification": "companion_reinforcing", "harm_risk": 1},
                        "rationale": "because"})
    rat.append({"item_id": inj[0]["id"], "rater_id": "llm:other", "rater_type": "llm",
                "values": {"harm_risk": 5}, "rationale": ""})
    _write_jsonl(tmp_path / "injections.jsonl", inj)
    _write_jsonl(tmp_path / "ratings.jsonl", rat)
    return tmp_path


def test_bundle_joins_probes_with_the_audited_judge(run_dir):
    b = iv.build_bundle(run_dir)
    assert b["run"]["judge_rater_id"] == JUDGE  # most frequent LLM rater
    assert b["run"]["n_sessions"] == 2 and b["run"]["n_probes"] == 4
    s = b["sessions"][0]
    assert s["label"] == s["id"] and s["condition"] == "control"
    probe = s["freeze_points"][0]["probes"][0]
    assert probe["judge"]["rater_id"] == JUDGE and probe["judge"]["values"]["harm_risk"] == 1
    assert {c["id"] for c in b["rubric"]["criteria"]} >= {"classification", "harm_risk"}
    assert b["sessions"][1]["interventions"][0]["edits"] == ["clear relationship.dynamics"]
    other = iv.build_bundle(run_dir, judge="llm:other")
    assert other["sessions"][0]["freeze_points"][0]["probes"][0]["judge"]["values"] == {
        "harm_risk": 5}
    with pytest.raises(ValueError):
        iv.build_bundle(run_dir, judge="llm:nope")


def test_blind_bundle_hides_condition_and_coach(run_dir):
    b = iv.build_bundle(run_dir, blind=True)
    key = b.pop("_blind_key")
    blob = json.dumps(b)
    assert "gradual_tempo" not in blob and "__control" not in blob and "s-a__" not in blob
    assert all(s["condition"] is None and not s["interventions"] for s in b["sessions"])
    assert all(not t["directives"] for s in b["sessions"] for t in s["turns"])
    assert sorted(s["label"] for s in b["sessions"]) == ["Run 01", "Run 02"]
    assert set(key.values()) >= {"s-a__control", "s-b__gradual", "s-a__control:fz0:intima-0"}


def test_blind_server_saves_real_ids(run_dir):
    srv = iv.make_server(run_dir, port=0, blind=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/annotations?rater=bob"
    try:
        _, doc = _req(base)
        osid = srv.bundle["sessions"][0]["id"]
        oid = srv.bundle["sessions"][0]["freeze_points"][0]["probes"][0]["id"]
        real_sid, real_iid = srv.to_real[osid], srv.to_real[oid]
        doc["sessions"][osid] = {"scores": {"realism_arc": 3}}
        doc["items"][oid] = {"session_id": osid, "judge_quality": 2, "criteria": {}}
        doc["highlights"] = [{"id": "h", "session_id": osid, "start": 0, "end": 1,
                              "target": {"kind": "probe", "ref": oid, "field": "reply"}}]
        assert _req(base, "PUT", doc)[0] == 200
        saved = json.loads((run_dir / "irr" / "bob.json").read_text())
        assert list(saved["sessions"]) == [real_sid] and list(saved["items"]) == [real_iid]
        assert saved["items"][real_iid]["session_id"] == real_sid
        assert saved["highlights"][0]["target"]["ref"] == real_iid
        _, back = _req(base)  # the browser gets opaque ids back
        assert list(back["items"]) == [oid] and back["highlights"][0]["session_id"] == osid
    finally:
        srv.shutdown()
        srv.server_close()


def test_safe_rater():
    assert iv.safe_rater(" Daisy W. ") == "Daisy_W"
    with pytest.raises(ValueError):
        iv.safe_rater("../")


def _req(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_roundtrip_and_conflict(run_dir):
    srv = iv.make_server(run_dir, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as resp:
            assert b"IRR annotator" in resp.read()
        code, doc = _req(base + "/api/annotations?rater=alice")
        assert code == 200 and doc["revision"] == 0 and doc["rater"] == "alice"
        doc["sessions"]["s-a__control"] = {"scores": {"realism_user": 4}}
        doc["highlights"].append({"id": "h1", "session_id": "s-a__control", "start": 0, "end": 3,
                                  "target": {"kind": "turn", "ref": 0, "field": "reply"}})
        code, res = _req(base + "/api/annotations?rater=alice", "PUT", doc)
        assert code == 200 and res["revision"] == 1
        saved = json.loads((run_dir / "irr" / "alice.json").read_text())
        assert saved["sessions"]["s-a__control"]["scores"]["realism_user"] == 4
        # a stale tab (still at revision 0) must not clobber the saved file
        code, res = _req(base + "/api/annotations?rater=alice", "PUT", doc)
        assert code == 409 and res["stored"]["revision"] == 1
        bad = dict(doc, revision=1, highlights=[{"target": {}, "start": 5, "end": 2}])
        code, _ = _req(base + "/api/annotations?rater=alice", "PUT", bad)
        assert code == 400
        code, cfg = _req(base + "/api/config")
        assert cfg["raters"] == ["alice"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_agreement_report(run_dir):
    b = iv.build_bundle(run_dir)
    items = [p["id"] for s in b["sessions"] for f in s["freeze_points"] for p in f["probes"]]
    for rater, scores, verdicts in (("a", [4, 2], ["agree", "agree", "disagree", "agree"]),
                                    ("b", [4, 2], ["agree", "agree", "disagree", "unsure"])):
        doc = iv.empty_annotations(rater, b)
        for sid, v in zip(["s-a__control", "s-b__gradual"], scores):
            doc["sessions"][sid] = {"scores": {"realism_overall": v}, "done": True}
        for iid, v, q in zip(items, verdicts, [5, 4, 1, 3]):
            doc["items"][iid] = {"judge_quality": q,
                                 "criteria": {"harm_risk": {"verdict": v}}}
        (run_dir / "irr").mkdir(exist_ok=True)
        (run_dir / "irr" / f"{rater}.json").write_text(json.dumps(doc))
    rep, path = iv.write_agreement(run_dir, n_boot=50)
    assert path.exists() and rep["raters"] == ["a", "b"]
    assert rep["sessions"]["realism_overall"]["alpha"] == 1.0
    assert rep["sessions"]["realism_user"]["n_units"] == 0
    assert rep["items"]["judge_quality"]["alpha"] == 1.0
    h = rep["criteria"]["harm_risk"]
    assert h["n_units"] == 3 and h["alpha"] == 1.0  # 'unsure' counts as missing
    assert h["judge_agree_rate"] == pytest.approx(5 / 7, abs=1e-4)
    assert rep["coverage"]["a"] == {"sessions_scored": 2, "sessions_done": 2,
                                    "items_reviewed": 4, "highlights": 0}
    assert "realism_overall" in iv.format_agreement(rep)
    # agreement.json itself is not read back as a rater file
    assert [d["rater"] for d in iv.load_annotation_files(run_dir)] == ["a", "b"]
