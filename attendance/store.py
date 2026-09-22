    # ---------- runtime settings (persisted per db; UI overrides env) ----------
    @property
    def settings_file(self) -> str:
        return os.path.join(self.cfg.db_dir, "settings.json")

    def get_settings(self) -> dict:
        with self._lock:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, encoding="utf-8") as f:
                    return json.load(f)
            return {}

    def set_setting(self, key: str, value) -> None:
        with self._lock:
            s = self.get_settings()
            s[key] = value
            _atomic_dump(self.settings_file, "w",
                         lambda f: json.dump(s, f, indent=1, ensure_ascii=False))

    # ---------- people admin ----------
    def ensure_person(self, person_id: str, name: str) -> bool:
        """Create a person record without face samples (batch import)."""
        with self._lock:
            people = self.load_people()
            if not person_id or person_id in people:
                return False
            people[person_id] = {"name": name, "samples": 0,
                                 "registered": self.cfg.now().isoformat(timespec="seconds")}
            self.save_people(people)
        return True

    def rename_person(self, person_id: str, new_name: str) -> bool:
        with self._lock:
            people = self.load_people()
            if person_id not in people or not new_name.strip():
                return False
            people[person_id]["name"] = new_name.strip()
            people[person_id]["updated"] = self.cfg.now().isoformat(timespec="seconds")
            self.save_people(people)
        return True

    def sample_counts(self) -> Dict[str, int]:
        enc = self.load_encodings()
        return {pid: len(v) for pid, v in enc.items()}

    def pending_samples(self, min_samples: Optional[int] = None) -> Dict[str, dict]:
        """People who still need photo samples (batch-import follow-up)."""
        need = min_samples or self.cfg.n_samples
        counts = self.sample_counts()
        return {pid: info for pid, info in self.load_people().items()
                if counts.get(pid, 0) < need}

    # ---------- manual status editing ----------
    def set_status(self, person_id: str, name: str, status: Optional[str],
                   class_id: str = "GENERAL", date: Optional[str] = None) -> bool:
        """Upsert a record with an explicit status. status=None removes the row
        (unmarks, so a later scan can re-mark)."""
        date = date or self._today()
        with self._lock:
            df = self._read_csv()
            mask = (df.Date == date) & (df.Class == class_id) & (df.ID == person_id)
            if status in ("Present", "Late"):
                if mask.any():
                    df.loc[mask, ["Name", "Status"]] = [name, status]
                else:
                    df = pd.concat([df, pd.DataFrame([{
                        "Date": date, "Time": self.cfg.now().strftime("%H:%M:%S"),
                        "Class": class_id, "ID": person_id, "Name": name,
                        "Status": status}])], ignore_index=True)
                self._marked.add((date, class_id, person_id))
            else:
                df = df[~mask]
                self._marked.discard((date, class_id, person_id))
            _atomic_dump(self.cfg.attendance_csv, "w",
                         lambda f: df.to_csv(f, index=False))
        self._sync_excel()
        self._backup()
        return True
