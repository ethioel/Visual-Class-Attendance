import itertools

import numpy as np

from attendance.config import Config
from attendance.store import Store

store = Store(Config())
enc = store.load_encodings()

genuine, impostor = [], []
for pid, samples in enc.items():
    for a, b in itertools.combinations(samples, 2):
        genuine.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
for (p1, s1), (p2, s2) in itertools.combinations(enc.items(), 2):
    for a in s1:
        for b in s2:
            impostor.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b))))

print(f"genuine  max: {max(genuine):.3f}")
print(f"impostor min: {min(impostor):.3f}")
print("Good tolerance sits between these two numbers.")
