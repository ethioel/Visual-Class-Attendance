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

COLS = ["Date", "Time", "Class", "ID", "Name", "Status"]


def _atomic_dump(path: str, mode: str, writer) -> None:
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
    """Thread-safe, crash-safe persistence: people, encodings, classes, attendance."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self._lock = threading.RLock()
        self._marked: Set[Tuple[str, str, str]] = set()   # (date, class, id)
        df = self._read_csv()
        if not df.empty:
            done = df[(df.Date == self._today()) & (df.Status.isin(["Present", "Late"]))]
            self._marked = {(self._today(), c, p)
                            for c, p in zip(done.Class, done.ID)}

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
            if person_id in people:
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

    # ---------- classes ----------
    @property
    def classes_file(self) -> str:
        return os.path.join(self.cfg.db_dir, "classes.json")

    def load_classes(self) -> Dict[str, dict]:
        with self._lock:
            if os.path.exists(self.classes_file):
                with open(self.classes_file, encoding="utf-8") as f:
                    return json.load(f)
            return {}

    def _save_classes(self, classes: Dict[str, dict]) -> None:
        with self._lock:
            _atomic_dump(self.classes_file, "w",
                         lambda f: json.dump(classes, f, indent=1, ensure_ascii=False))

    def create_class(self, class_id: str, name: str, teacher: str,
                     late_after: Optional[str] = None) -> bool:
        classes = self.load_classes()
        class_id = class_id.strip()
        if not class_id or class_id in classes:
            return False
        classes[class_id] = {"name": name, "teacher": teacher,
                             "late_after": late_after or None,
                             "students": [],
                             "created": self.cfg.now().isoformat(timespec="seconds")}
        self._save_classes(classes)
        return True

    def delete_class(self, class_id: str) -> bool:
        classes = self.load_classes()
        if class_id not in classes:
            return False
        classes.pop(class_id)
        self._save_classes(classes)
        return True

    def update_class(self, class_id: str, changes: Dict) -> bool:
        classes = self.load_classes()
        if class_id not in classes:
            return False
        for k, v in changes.items():
            if k == "late_after":
                classes[class_id]["late_after"] = v or None
            elif k in ("name", "teacher"):
                classes[class_id][k] = v
        self._save_classes(classes)
        return True

    def add_to_class(self, class_id: str, person_id: str) -> bool:
        classes = self.load_classes()
        if class_id not in classes or person_id in classes[class_id]["students"]:
            return False
        classes[class_id]["students"].append(person_id)
        self._save_classes(classes)
        return True

    def remove_from_class(self, class_id: str, person_id: str) -> bool:
        classes = self.load_classes()
        if class_id not in classes or person_id not in classes[class_id]["students"]:
            return False
        classes[class_id]["students"].remove(person_id)
        self._save_classes(classes)
        return True

    def classes_of(self, username: str) -> Dict[str, dict]:
        return {cid: c for cid, c in self.load_classes().items()
                if c.get("teacher") == username}

    # ---------- attendance ----------
    def _today(self) -> str:
        return self.cfg.now().strftime("%Y-%m-%d")

    def status_now(self, late_after: Optional[str] = None) -> str:
        threshold = late_after or self.cfg.late_after
        if not threshold:
            return "Present"
        return "Late" if self.cfg.now().strftime("%H:%M") > threshold else "Present"

    def _read_csv(self) -> pd.DataFrame:
        if os.path.exists(self.cfg.attendance_csv):
            df = pd.read_csv(self.cfg.attendance_csv, dtype=str).fillna("")
            if "Class" not in df.columns:      # migrate pre-v2.1 logs
                df["Class"] = "GENERAL"
            return df
        return pd.DataFrame(columns=COLS)

    def records_df(self, class_id: Optional[str] = None) -> pd.DataFrame:
        with self._lock:
            df = self._read_csv()
        if class_id and not df.empty:
            df = df[df.Class == class_id]
        return df

    def today_df(self, class_id: Optional[str] = None) -> pd.DataFrame:
        df = self.records_df(class_id)
        return df[df.Date == self._today()].sort_values("Time") if not df.empty else df

    def mark(self, person_id: str, name: str, status: Optional[str] = None,
             class_id: str = "GENERAL") -> bool:
        today, status = self._today(), (status or self.status_now())
        now_t = self.cfg.now().strftime("%H:%M:%S")
        with self._lock:
            if (today, class_id, person_id) in self._marked:
                return False
            df = self._read_csv()
            if not df.empty and ((df.Date == today) & (df.Class == class_id)
                                 & (df.ID == person_id)).any():
                self._marked.add((today, class_id, person_id))
                return False
            new_file = not os.path.exists(self.cfg.attendance_csv) or \
                       os.path.getsize(self.cfg.attendance_csv) == 0
            with open(self.cfg.attendance_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(COLS)
                w.writerow([today, now_t, class_id, person_id, name, status])
                f.flush()
                os.fsync(f.fileno())
            self._marked.add((today, class_id, person_id))
        self._sync_excel()
        self._backup()
        return True

    def _present_today(self, class_id: str) -> Set[str]:
        df = self.records_df(class_id)
        if df.empty:
            return set()
        m = ((df.Date == self._today()) & df.Status.isin(["Present", "Late"]))
        return set(df[m].ID)

    def mark_absent_all(self, class_id: Optional[str] = None) -> int:
        """Absent per roster. class_id=None → every class, plus unassigned
        people as GENERAL (keeps the CLI flow working without classes)."""
        people, classes = self.load_people(), self.load_classes()
        if class_id:
            targets = {class_id: classes.get(class_id, {"students": []})}
        else:
            targets = dict(classes)
            in_any = {p for c in classes.values() for p in c.get("students", [])}
            leftovers = [pid for pid in people if pid not in in_any]
            if leftovers:
                targets["GENERAL"] = {"students": leftovers}
        count = 0
        for cid, cls in targets.items():
            present = self._present_today(cid)
            for pid in cls.get("students", []):
                if pid not in present and self.mark(
                        pid, people.get(pid, {}).get("name", pid), "Absent", cid):
                    count += 1
        return count

    def attendance_rates(self, class_id: str) -> pd.DataFrame:
        """Sessions = dates on which this class has any record."""
        df = self.records_df(class_id)
        people = self.load_people()
        roster = self.load_classes().get(class_id, {}).get("students", [])
        if df.empty or not roster:
            return pd.DataFrame(columns=["ID", "Name", "Sessions", "Attended", "Rate"])
        sessions = df.Date.nunique()
        attended = df[df.Status.isin(["Present", "Late"])].groupby("ID").Date.nunique()
        rows = [{"ID": pid, "Name": people.get(pid, {}).get("name", pid),
                 "Sessions": sessions, "Attended": int(attended.get(pid, 0)),
                 "Rate": f"{100 * attended.get(pid, 0) / sessions:.0f}%"}
                for pid in roster]
        return pd.DataFrame(rows).sort_values("Rate")

    def _sync_excel(self) -> None:
        try:
            df = self.records_df()
            _atomic_dump(self.cfg.attendance_xlsx, "wb",
                         lambda f: df.to_excel(f, index=False, engine="openpyxl"))
        except Exception:
            pass

    def _backup(self) -> None:
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
