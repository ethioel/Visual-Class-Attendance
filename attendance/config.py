from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Config:
    db_dir: str = field(default_factory=lambda: os.environ.get("ATT_DB_DIR", "attendance_db"))
    tolerance: float = field(default_factory=lambda: float(os.environ.get("ATT_TOLERANCE", "0.55")))
    n_samples: int = field(default_factory=lambda: int(os.environ.get("ATT_N_SAMPLES", "6")))
    detect_scale: float = field(default_factory=lambda: float(os.environ.get("ATT_DETECT_SCALE", "0.5")))
    frame_skip: int = field(default_factory=lambda: int(os.environ.get("ATT_FRAME_SKIP", "3")))
    late_after: Optional[str] = field(default_factory=lambda: os.environ.get("ATT_LATE_AFTER", "09:00") or None)
    timezone: str = field(default_factory=lambda: os.environ.get("ATT_TZ", ""))  # e.g. "Africa/Addis_Ababa"
    camera_index: int = field(default_factory=lambda: int(os.environ.get("ATT_CAMERA", "0")))
    enroll_jitters: int = field(default_factory=lambda: int(os.environ.get("ATT_ENROLL_JITTERS", "10")))
    
    @property
    def attendance_csv(self) -> str: return os.path.join(self.db_dir, "attendance_log.csv")
    @property
    def attendance_xlsx(self) -> str: return os.path.join(self.db_dir, "attendance.xlsx")
    @property
    def enc_cache(self) -> str: return os.path.join(self.db_dir, "encodings.pkl")
    @property
    def people_file(self) -> str: return os.path.join(self.db_dir, "people.json")
    @property
    def unknown_dir(self) -> str: return os.path.join(self.db_dir, "unknown")
    @property
    def snapshot_dir(self) -> str: return os.path.join(self.db_dir, "snapshots")

    def ensure_dirs(self) -> None:
        for d in (self.db_dir, self.unknown_dir, self.snapshot_dir):
            os.makedirs(d, exist_ok=True)

    def now(self) -> datetime:
        if self.timezone:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(self.timezone))
        return datetime.now()