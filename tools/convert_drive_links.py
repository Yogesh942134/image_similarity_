import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "..", "inference", "faiss", "paths.npy")

paths = np.load(PATH, allow_pickle=True)

new = []
for p in paths:
    fid = p.split("id=")[-1]
    new.append(f"https://drive.google.com/thumbnail?id={fid}&sz=w1000")

np.save(PATH, np.array(new))
print("Drive URLs converted successfully!")
