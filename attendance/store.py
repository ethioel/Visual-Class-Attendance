from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
import threading
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import Config

COLS = ["Date", "Time", "ID", "Name", "Status"]


def _atomic_dump(path: str, mode: str, writer) -> None:
    """Temp file in same dir → fsync → atomic replace (crash-safe)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class Store:
    """Thread-safe, crash-safe persistence for people, encodings and attendance."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self._lock = threading.RLock()
        self._marked: Set[Tuple[str, str]] = set()          # (date, id) this process
        df = self._read_csv()
        if not df.empty:
            done = df[(df.Date == self._today()) & (df.Status.isin(["Present", "Late"]))]
            self._marked = {(self._today(), pid) for pid in done.ID}

    # ---------- people ----------
    def load_people(self) -> Dict[str, dict]:
        with self._lock:
            if os.path.exists(self.cfg.people_file):
                with open(self.cfg.people_file, encoding="utf-8") as f:
                    return json.load(f)
            return {}

    def save_people(self, people: Dict[str, dict]) -> None:
        with self._lock:
            _atomic_dump(self.cfg.people_file, "w",
                         lambda f: json.dump(people, f, indent=1, ensure_ascii=False))

    def enroll(self, person_id: str, name: str, encodings: List[np.ndarray]) -> None:
        with self._lock:
            people, enc = self.load_people(), self.load_encodings()
            if person_id in people:                       # append samples to existing
                enc[person_id] = list(enc.get(person_id, [])) + list(encodings)
                people[person_id]["samples"] = len(enc[person_id])
                people[person_id]["updated"] = self.cfg.now().isoformat(timespec="seconds")
            else:
                enc[person_id] = list(encodings)
                people[person_id] = {"name": name, "samples": len(encodings),
                                     "registered": self.cfg.now().isoformat(timespec="seconds")}
            self.save_people(people)
            self.save_encodings(enc)

    def remove_person(self, person_id: str) -> bool:
        with self._lock:
            people = self.load_people()
            if person_id not in people:
                return False
            people.pop(person_id)
            self.save_people(people)
            enc = self.load_encodings()
            enc.pop(person_id, None)
            self.save_encodings(enc)
        return True

    # ---------- encodings ----------
    def load_encodings(self) -> Dict[str, List[np.ndarray]]:
        with self._lock:
            people = self.load_people()
            if not os.path.exists(self.cfg.enc_cache):
                return {}
            with open(self.cfg.enc_cache, "rb") as f:
                enc = pickle.load(f)
            return {k: v for k, v in enc.items() if k in people}

    def save_encodings(self, enc: Dict[str, List[np.ndarray]]) -> None:
        with self._lock:
            _atomic_dump(self.cfg.enc_cache, "wb", lambda f: pickle.dump(enc, f))

    # ---------- attendance ----------
    def _today(self) -> str:
        return self.cfg.now().strftime("%Y-%m-%d")

    def status_now(self) -> str:
        if not self.cfg.late_after:
            return "Present"
        return "Late" if self.cfg.now().strftime("%H:%M") > self.cfg.late_after else "Present"

    def _read_csv(self) -> pd.DataFrame:
        if os.path.exists(self.cfg.attendance_csv):
            return pd.read_csv(self.cfg.attendance_csv, dtype=str).fillna("")
        return pd.DataFrame(columns=COLS)

    def records_df(self) -> pd.DataFrame:
        with self._lock:
            return self._read_csv()

    def today_df(self) -> pd.DataFrame:
        df = self.records_df()
        return df[df.Date == self._today()].sort_values("Time") if not df.empty else df

    def mark(self, person_id: str, name: str, status: Optional[str] = None) -> bool:
        """Append one record. False if this person already has a record today."""
        today, status = self._today(), (status or self.status_now())
        now_t = self.cfg.now().strftime("%H:%M:%S")
        with self._lock:
            if (today, person_id) in self._marked:
                return False
            df = self._read_csv()                          # cross-check file too
            if not df.empty and ((df.Date == today) & (df.ID == person_id)).any():
                self._marked.add((today, person_id))
                return False
            new_file = not os.path.exists(self.cfg.attendance_csv) or \
                       os.path.getsize(self.cfg.attendance_csv) == 0
            with open(self.cfg.attendance_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(COLS)
                w.writerow([today, now_t, person_id, name, status])
                f.flush()
                os.fsync(f.fileno())                       # crash-safe append
            self._marked.add((today, person_id))
        self._sync_excel()
        self._backup()
        return True

    def mark_absent_all(self) -> int:
        people, today, df = self.load_people(), self._today(), self.records_df()
        present = set(df[(df.Date == today) & (df.Status.isin(["Present", "Late"]))].ID) \
            if not df.empty else set()
        return sum(1 for pid, info in people.items()
                   if pid not in present and self.mark(pid, info.get("name", pid), "Absent"))

    def _sync_excel(self) -> None:
        try:
            df = self.records_df()
            _atomic_dump(self.cfg.attendance_xlsx, "wb",
                         lambda f: df.to_excel(f, index=False, engine="openpyxl"))
        except Exception:
            pass                                           # CSV is source of truth

    def _backup(self) -> None:
        """Optional: push CSV to a HF Dataset repo on every record (set secrets)."""
        repo, token = os.environ.get("HF_DATASET_REPO"), os.environ.get("HF_TOKEN")
        if not (repo and token):
            return
        try:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(
                path_or_fileobj=self.cfg.attendance_csv, path_in_repo="attendance_log.csv",
                repo_id=repo, repo_type="dataset", commit_message="attendance backup")
        except Exception:
            pass