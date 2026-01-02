import numpy as np, os, re

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "..", "inference", "faiss", "paths.npy")

paths = np.load(PATH, allow_pickle=True)

clean = []

for p in paths:
    # extract only the real Drive file ID
    m = re.search(r'([a-zA-Z0-9_-]{20,})', p)
    if not m:
        continue
    fid = m.group(1)
    clean.append(f"https://lh3.googleusercontent.com/d/{fid}=w600")

np.save(PATH, np.array(clean))
print("paths.npy rebuilt cleanly!")
