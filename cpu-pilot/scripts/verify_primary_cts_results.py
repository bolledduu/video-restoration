"""Independently check a completed primary study against its saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(job):
    protocol = read(job / "source_protocol.json")
    config = read(job / "config.json")
    frozen_path = job / "frozen_selector.json"
    frozen = read(frozen_path)
    result = read(job / "TEST_RESULTS.json")
    audit_path = job / "test/audit/measurements.json"
    audit = read(audit_path)
    raw = read(job / "test/raw/measurements.json")
    events = [json.loads(line) for line in (job / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = protocol["splits"]
    assert len(set(splits["train"]) | set(splits["calibration"]) | set(splits["test"])) == sum(map(len, splits.values()))
    assert sha(job / "trained.json") == frozen["trained_sha256"]
    assert sha(job / "calibration/audit/measurements.json") == frozen["calibration_sha256"]
    assert sha(frozen_path) == result["frozen_selector_sha256"]
    assert sha(audit_path) == result["test_audit_sha256"]
    assert audit["status"] == "COMPLETE" and raw["status"] in ("COMPLETE", "COMPLETE_WITH_EXCLUSIONS")
    assert audit["weights"]["talknet"] == config["checkpoint_sha256"]
    first_test = min(e["updated_unix"] for e in events if e["phase"] == "test_raw")
    freeze = max(e["updated_unix"] for e in events if e["phase"] == "freeze_calibration")
    assert freeze < first_test
    assert all(e.get("frozen_selector_sha256") == sha(frozen_path) for e in events if e["phase"] in ("test_raw", "test_audit", "resume_frozen_test") and "frozen_selector_sha256" in e)
    cases = {c["id"]: c for c in audit["cases"]}
    raw_cases = {c["id"]: c for c in raw["cases"]}
    assert len(cases) == len(audit["cases"]) == len(raw_cases) == result["case_count"]
    assert all(c["source"] in splits["test"] for c in cases.values())
    assert result["source_count"] == len({c["source"] for c in cases.values()})
    assert len(raw_cases) + len(raw["exclusions"]) == raw["expected_episodes"]
    assert result["test_exclusions"] == raw["exclusions"]
    for case_id, c in cases.items():
        raw_case = raw_cases[case_id]
        assert c["labels"] == raw_case["labels"]
        assert c["captured_probability"] == raw_case["candidates"]["captured"]["probability"]
        assert c["restored_probability"] == raw_case["candidates"]["fastdvdnet"]["probability"]
    for method, rows in result["rows"].items():
        assert len(rows) == len(cases) and {r["case_id"] for r in rows} == set(cases)
        for row in rows:
            c = cases[row["case_id"]]
            labels = c["labels"]
            captured = [p >= .5 for p in c["captured_probability"]]
            restored = [p >= .5 for p in c["restored_probability"]]
            selected = restored if row["selected"] == "restored" else captured
            assert row["source"] == c["source"] and row["eligible"] == len(labels)
            assert row["captured_errors"] == sum(p != bool(y) for p, y in zip(captured, labels))
            assert row["errors"] == sum(p != bool(y) for p, y in zip(selected, labels))
            assert row["harm"] == sum(cp == bool(y) and sp != bool(y) for cp, sp, y in zip(captured, selected, labels))
            assert row["corrections"] == sum(cp != bool(y) and sp == bool(y) for cp, sp, y in zip(captured, selected, labels))
        summary = result["summaries"][method]
        for field in ("eligible", "errors", "captured_errors", "harm", "corrections"):
            assert summary[field] == sum(r[field] for r in rows)
        assert summary["restored_cases"] == sum(r["selected"] == "restored" for r in rows)
        assert abs(summary["mean_delta_musiq"] - sum(r["delta_musiq"] for r in rows) / len(rows)) < 1e-10
    return {"status": "VERIFIED", "sources": result["source_count"], "cases": result["case_count"], "excluded": len(raw["exclusions"]), "frozen_selector_sha256": sha(frozen_path), "result_sha256": sha(job / "TEST_RESULTS.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.job), indent=2))
