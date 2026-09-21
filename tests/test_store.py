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

def test_absent_flow(tmp_path):
    s = make_store(tmp_path)
    s.enroll("X1", "Alice", [np.zeros(128)]); s.enroll("X2", "Bob", [np.ones(128)])
    s.mark("X1", "Alice")
    assert s.mark_absent_all() == 1
    assert set(s.records_df().Status) == {"Present", "Absent"}