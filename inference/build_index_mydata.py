import torch, faiss, numpy as np, os, csv, requests
from torchvision import transforms
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from models.embedding_model import EmbeddingNet
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "drive_urls.csv")
MODEL = os.path.join(BASE, "..", "models", "embedding.pth")
FAISS_DIR = os.path.join(BASE, "faiss")
os.makedirs(FAISS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()

with open(CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("Total Drive images:", len(rows))

batch_size = 64
embeddings, paths = [], []

def fetch(url):
    try:
        return requests.get(url, timeout=10).content
    except:
        return None

for i in tqdm(range(0, len(rows), batch_size)):
    batch = rows[i:i+batch_size]
    urls = [r["url"] for r in batch]

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch, urls))

    imgs = []
    for data in results:
        if data is None:
            continue
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            imgs.append(transform(img))
        except:
            continue  # skip corrupted / non-image files

    if not imgs:
        continue

    imgs = torch.stack(imgs).to(device)

    with torch.no_grad():
        embs = model(imgs).cpu().numpy().astype("float32")
        faiss.normalize_L2(embs)

    embeddings.append(embs)
    paths.extend(urls[:len(embs)])

embeddings = np.vstack(embeddings)

index = faiss.IndexFlatIP(128)
index.add(embeddings)

faiss.write_index(index, os.path.join(FAISS_DIR, "index.bin"))
np.save(os.path.join(FAISS_DIR, "paths.npy"), np.array(paths))

print("Drive FAISS index built successfully!")
