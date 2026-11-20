"""A certified interval is evidence about one system, student and certificate."""

from __future__ import annotations

from frost.certified import Certified, lookup, read, record

SI = {"run": "si512_T300", "student": "si512_T300_s35c16e1", "period_fs": 78.7,
      "epsilon": 2.0, "gain_budget": 1.0}


def entry(**over) -> Certified:
    fields = {**SI, "k": 10, "ceiling": 10, "evidence_fs": 139_626.0, "refusals": 3}
    fields.update(over)
    return Certified(**fields)


def test_a_ceiling_is_returned_only_for_the_same_system_student_and_certificate(tmp_path):
    path = tmp_path / "certified.json"
    record(path, entry())
    assert lookup(path, **SI).ceiling == 10
    for changed in ({"run": "cu500_T300"}, {"student": "si512_T300_m45c16"},
                    {"period_fs": 27.7}, {"epsilon": 1.0}, {"gain_budget": 0.5}):
        assert lookup(path, **{**SI, **changed}) is None, changed


def test_a_ceiling_that_was_never_held_is_not_certified(tmp_path):
    """Si's 60 ps run ended mid-descent: it reached k = 12 without certifying it."""
    path = tmp_path / "certified.json"
    record(path, entry(k=12, ceiling=12, evidence_fs=19_660.0, refusals=2))
    assert lookup(path, **SI) is None  # under one min_ps of evidence
    assert lookup(path, **SI, min_hold_fs=10_000.0).ceiling == 12


def test_re_recording_replaces_rather_than_accumulates(tmp_path):
    path = tmp_path / "certified.json"
    record(path, entry(ceiling=12, evidence_fs=60_000.0))
    record(path, entry(ceiling=10, evidence_fs=139_626.0))
    assert lookup(path, **SI).ceiling == 10
    assert len(read(path)) == 1


def test_a_thin_margin_is_not_inherited_whole():
    """Si's k = 12 held for 160 ps on one seed and was refused on another."""
    thin = entry(k=12, ceiling=12, projection=0.72)  # 0.72 * 1.5 > 1.0
    assert thin.start_ceiling() == 10  # re-earn the boundary rung
    comfortable = entry(k=10, ceiling=10, projection=0.53)  # 0.53 * 1.5 < 1.0
    assert comfortable.start_ceiling() == 10  # inherited whole


def test_an_entry_without_a_recorded_projection_is_inherited_as_before():
    assert entry(ceiling=12, projection=0.0).start_ceiling() == 12


def test_the_refusal_sequence_survives_a_round_trip_and_is_not_identity(tmp_path):
    """Two Si seeds differ by one rung at one timestamp; the count cannot show that."""
    path = tmp_path / "certified.json"
    seed1 = entry(k=10, ceiling=10, projection=0.53, refusals=3, refused=[
        {"k": 16, "t_fs": 20314.0, "projection": 2.03},
        {"k": 14, "t_fs": 40330.0, "projection": 1.42},
        {"k": 12, "t_fs": 60340.0, "projection": 1.04}])
    record(path, seed1)
    back = lookup(path, **SI)
    assert [row["k"] for row in back.refused] == [16, 14, 12]
    assert back.start_ceiling() == 10

    # evidence, not identity: re-recording with a different sequence replaces the entry
    record(path, entry(k=12, ceiling=12, projection=0.72, refusals=2, refused=[
        {"k": 16, "t_fs": 20314.0, "projection": 2.03},
        {"k": 14, "t_fs": 40330.0, "projection": 1.42}]))
    assert len(read(path)) == 1
    assert lookup(path, **SI).start_ceiling() == 10  # thin margin, demoted


def test_entries_written_before_the_field_read_back_empty(tmp_path):
    import json as _json

    path = tmp_path / "certified.json"
    record(path, entry())
    rows = _json.loads(path.read_text())
    del rows[0]["refused"]  # as an older writer would have left it
    path.write_text(_json.dumps(rows))
    assert lookup(path, **SI).refused == []
