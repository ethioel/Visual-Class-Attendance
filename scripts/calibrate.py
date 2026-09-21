# scripts/calibrate.py
import itertools
import numpy as np

from attendance.config import Config
from attendance.store import Store

store = Store(Config())        # or Config(db_dir="path/to/data") if run elsewhere
enc = store.load_encodings()

genuine, impostor = [], []
for pid, samples in enc.items():
    for a, b in itertools.combinations(samples, 2):
        genuine.append(np.linalg.norm(a - b))            # same person, different photos
for (p1, s1), (p2, s2) in itertools.combinations(enc.items(), 2):
    for a in s1:
        for b in s2:
            impostor.append(np.linalg.norm(a - b))       # different people

print(f"genuine  max: {max(genuine):.3f}")
print(f"impostor min: {min(impostor):.3f}")
# Good threshold sits between those two numbers. If they overlap → add more/better samples.