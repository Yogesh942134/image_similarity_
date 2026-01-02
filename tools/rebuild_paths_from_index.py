import numpy as np, os

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "..", "inference", "faiss", "paths.npy")

paths = np.load(PATH, allow_pickle=True)

# FAISS index size
import faiss
INDEX = os.path.join(BASE, "..", "inference", "faiss", "index.bin")
index = faiss.read_index(INDEX)

paths = paths[:index.ntotal]

np.save(PATH, np.array(paths))
print("paths.npy trimmed to match FAISS index!")
