from datetime import datetime

import numpy as np

from attendance.config import Config
from attendance.store import Store


def make_store(tmp_path):
    return Store(Config(db_dir=str(tmp_path), late_after=None))


def test_mark_and_duplicate(tmp_path):
    s = make_store(tmp_path)
    assert s.mark("X1", "Alice") is True
    assert s.mark("X1", "Alice") is False
    assert len(s.records_df()) == 1


def test_same_person_two_classes_same_day(tmp_path):
    s = make_store(tmp_path)
    s.create_class("c1", "A", "t"); s.create_class("c2", "B", "t")
    assert s.mark("X1", "Alice", class_id="c1")
    assert s.mark("X1", "Alice", class_id="c2")
    assert len(s.records_df()) == 2


def test_absent_flow_unassigned(tmp_path):
    s = make_store(tmp_path)
    s.enroll("X1", "Alice", [np.zeros(128)]); s.enroll("X2", "Bob", [np.ones(128)])
    s.mark("X1", "Alice")
    assert s.mark_absent_all() == 1
    assert set(s.records_df().Status) == {"Present", "Absent"}


def test_class_roster_absent(tmp_path):
    s = make_store(tmp_path)
    s.create_class("c1", "CS", "t", late_after="09:00")
    s.enroll("X1", "Alice", [np.zeros(128)]); s.enroll("X2", "Bob", [np.ones(128)])
    s.add_to_class("c1", "X1")
    assert s.mark("X1", "Alice", class_id="c1")
    assert s.mark_absent_all("c1") == 0
    s.add_to_class("c1", "X2")
    assert s.mark_absent_all("c1") == 1


def test_excused_survives_absent_and_rates(tmp_path):
    s = make_store(tmp_path)
    s.create_class("c1", "CS", "t")
    s.enroll("X1", "Alice", [np.zeros(128)]); s.enroll("X2", "Bob", [np.ones(128)])
    s.add_to_class("c1", "X1"); s.add_to_class("c1", "X2")
    s.mark("X1", "Alice", "Present", "c1")
    s.set_status("X2", "Bob", "Excused", "c1")
    assert s.mark_absent_all("c1") == 0              # excused untouched
    r = s.attendance_rates("c1").set_index("ID")
    assert r.loc["X1", "Rate"] == "100%"
    assert r.loc["X2", "Rate"] == "—"                # all sessions excused


def test_set_status_upsert_and_unmark(tmp_path):
    s = make_store(tmp_path)
    s.mark("X1", "Alice", "Present")
    s.set_status("X1", "Alice", "Late")              # edit upserts
    assert s.records_df().iloc[0]["Status"] == "Late"
    s.set_status("X1", "Alice", None)                # unmark removes
    assert s.records_df().empty
    assert s.mark("X1", "Alice") is True             # re-markable after unmark


def test_remove_person_strips_rosters(tmp_path):
    s = make_store(tmp_path)
    s.create_class("c1", "CS", "t")
    s.enroll("X1", "Alice", [np.zeros(128)])
    s.add_to_class("c1", "X1")
    assert s.remove_person("X1")
    assert s.load_classes()["c1"]["students"] == []


def test_legacy_csv_migration(tmp_path):
    cfg = Config(db_dir=str(tmp_path), late_after=None)
    cfg.ensure_dirs()
    today = datetime.now().strftime("%Y-%m-%d")
    with open(cfg.attendance_csv, "w") as f:
        f.write(f"Date,Time,ID,Name,Status\n{today},08:00:00,OLD,Old,Present\n")
    s = Store(cfg)
    df = s.records_df()
    assert "Class" in df.columns and df.iloc[0]["Class"] == "GENERAL"
    assert s.mark("OLD", "Old", class_id="GENERAL") is False
