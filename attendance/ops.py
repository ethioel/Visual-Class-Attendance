from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class OpsStore:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()
        self._lock = threading.RLock()
        self.path = os.path.join(cfg.db_dir, "ops.json")

    # ---------- storage ----------
    def _load(self) -> dict:
        with self._lock:
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            return {"timetables": {}, "rooms": {}, "policy": {}}

    def _save(self, data: dict) -> None:
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path),
                                       suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

    def _audit(self, actor: str, action: str, detail: str = "") -> None:
        """Same append-only audit.csv the attendance Store writes."""
        try:
            path = os.path.join(self.cfg.db_dir, "audit.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["Time", "Actor", "Action", "Detail"])
                w.writerow([self.cfg.now().isoformat(timespec="seconds"),
                            actor, action, detail])
        except Exception:
            pass

    # ======================= policy =======================
    def policy(self) -> dict:
        p = self._load()["policy"]
        return {"min_rate": float(p.get("min_rate", 75)),      # % to sit exam
                "stale_days": int(p.get("stale_days", 7)),     # no-session flag
                "manual_flag": int(p.get("manual_flag", 3)),   # manual marks
                "workdays": p.get("workdays", DAYS[:6])}

    def set_policy(self, actor: str, **changes) -> None:
        data = self._load()
        data["policy"].update(changes)
        self._save(data)
        self._audit(actor, "set_policy", str(changes))

    # ======================= rooms =======================
    def set_room(self, cid: str, room: str) -> None:
        data = self._load()
        if room.strip():
            data["rooms"][cid] = room.strip()
        else:
            data["rooms"].pop(cid, None)
        self._save(data)

    def room_of(self, cid: str) -> str:
        return self._load()["rooms"].get(cid, "—")

    # ======================= timetable =======================
    def set_slots(self, actor: str, cid: str, slots: List[dict]) -> None:
        """slots: [{day:'Mon', start:'09:00', end:'10:30'}]"""
        data = self._load()
        data["timetables"][cid] = sorted(
            slots, key=lambda s: (DAYS.index(s["day"]) if s["day"] in DAYS
                                  else 99, s.get("start", "")))
        self._save(data)
        self._audit(actor, "set_timetable", f"{cid} {len(slots)} slots")

    def slots_of(self, cid: str) -> List[dict]:
        return self._load()["timetables"].get(cid, [])

    def now_and_next(self, cid: str) -> tuple:
        """(now_slot, next_slot) for this class vs current time, using the
        configured timezone. (None, None) if no timetable."""
        import datetime as dt
        slots = self.slots_of(cid)
        if not slots:
            return None, None
        now = self.cfg.now()
        today = DAYS[now.weekday()]
        t = now.strftime("%H:%M")
        todays = [s for s in slots if s["day"] == today]
        for s in todays:                                   # happening now
            if s["start"] <= t <= s["end"]:
                return s, None
        for s in todays:                                   # later today
            if s["start"] > t:
                return None, s
        for i in range(1, 8):                              # next few days
            d = DAYS[(now.weekday() + i) % 7]
            nxt = [s for s in slots if s["day"] == d]
            if nxt:
                return None, nxt[0]
        return None, None

    # ======================= oversight analytics =======================
    def oversight(self, store, classes: Dict[str, dict],
                  people: Dict[str, dict]) -> pd.DataFrame:
        """One row per class: sessions, last session, avg rate, flags."""
        pol = self.policy()
        enc = store.load_encodings()
        rows = []
        for cid, cls in classes.items():
            df = store.records_df(cid)
            roster = cls.get("students", [])
            sessions = int(df.Date.nunique()) if not df.empty else 0
            last = str(df.Date.max()) if not df.empty else ""
            rates = store.attendance_rates(cid)
            if not rates.empty and "Rate" in rates:
                rates = rates[rates.Rate != "—"]
                vals = pd.to_numeric(rates.Rate.astype(str).str.rstrip("%"),
                                     errors="coerce").dropna()
                avg = f"{vals.mean():.0f}%" if not vals.empty else "—"
            else:
                avg = "—"
            flags = []
            if not roster:
                flags.append("empty roster")
            if sessions == 0:
                flags.append("no sessions yet")
            elif last and (date.today() - date.fromisoformat(last)).days \
                    > pol["stale_days"]:
                flags.append(f"no session in {pol['stale_days']}+ days")
            if avg != "—" and roster and \
                    float(avg.rstrip("%")) < pol["min_rate"]:
                flags.append(f"avg below {pol['min_rate']}%")
            rows.append({
                "Class": cls.get("name", cid), "ID": cid,
                "Teacher": cls.get("teacher", "—"),
                "Room": self.room_of(cid),
                "Students": len(roster),
                "Ready": sum(1 for p in roster if p in enc),
                "Sessions": sessions,
                "Last session": last or "—",
                "Avg rate": avg,
                "Flags": " · ".join(flags) if flags else "✅"})
        return pd.DataFrame(rows)

    def defaulters(self, store, cid: str, people: Dict[str, dict]) -> pd.DataFrame:
        """Students below the policy minimum (excused-adjusted rates)."""
        pol = self.policy()
        r = store.attendance_rates(cid)
        if r.empty:
            return pd.DataFrame(columns=["ID", "Name", "Rate", "Reason"])
        r = r.copy()
        r["pct"] = pd.to_numeric(r.Rate.astype(str).str.rstrip("%"),
                                 errors="coerce")
        d = r[r.pct < pol["min_rate"]].copy()
        if d.empty:
            return pd.DataFrame(columns=["ID", "Name", "Rate", "Reason"])
        d["Name"] = d.ID.map(lambda p: people.get(p, {}).get("name", p))
        d["Reason"] = d.pct.map(
            lambda v: f"below {pol['min_rate']}% minimum ({v:.0f}%)")
        return d[["ID", "Name", "Sessions", "Attended", "Rate", "Reason"]]

    def anomalies(self, store, audit_df: pd.DataFrame) -> pd.DataFrame:
        """Students frequently marked manually — usually recognition failing
        for them (bad samples) or proxy attempts. Both need attention."""
        pol = self.policy()
        if audit_df is None or audit_df.empty:
            return pd.DataFrame(columns=["ID", "Manual marks", "Hint"])
        m = audit_df[audit_df.Action.isin(["mark_manual", "set_status"])]
        if m.empty:
            return pd.DataFrame(columns=["ID", "Manual marks", "Hint"])
        counts = (m.Detail.str.extract(r"/([^=]+)=")[0]
                  .dropna().value_counts())
        rows = [{"ID": pid, "Manual marks": int(n),
                 "Hint": ("recognition likely failing — re-enroll samples"
                          if n >= pol["manual_flag"]
                          else "occasional manual marks")}
                for pid, n in counts.items() if n >= 2]
        return pd.DataFrame(rows).sort_values(
            "Manual marks", ascending=False) if rows else \
            pd.DataFrame(columns=["ID", "Manual marks", "Hint"])

    def institutional_report(self, store, classes: Dict[str, dict],
                             people: Dict[str, dict],
                             month: str) -> pd.DataFrame:
        """Per-student rows for one month across all classes — the
        exportable institutional report."""
        pol = self.policy()
        rows = []
        for cid, cls in classes.items():
            df = store.records_df(cid)
            if df.empty:
                continue
            df = df[df.Date.str.startswith(month)]
            if df.empty:
                continue
            sessions = int(df.Date.nunique())
            attended = df[df.Status.isin(["Present", "Late"])].groupby(
                "ID").Date.nunique()
            excused = df[df.Status == "Excused"].groupby("ID").Date.nunique()
            for pid in cls.get("students", []):
                eff = sessions - int(excused.get(pid, 0))
                att = int(attended.get(pid, 0))
                rate = f"{100 * att / eff:.0f}%" if eff > 0 else "—"
                sits = (eff > 0 and att / eff >= pol["min_rate"] / 100)
                rows.append({
                    "Month": month, "Class": cls.get("name", cid),
                    "ID": pid,
                    "Name": people.get(pid, {}).get("name", pid),
                    "Sessions": sessions, "Attended": att,
                    "Excused": int(excused.get(pid, 0)),
                    "Rate": rate,
                    f"Meets {int(pol['min_rate'])}%": "YES" if sits else "NO"})
        return pd.DataFrame(rows)
