import faiss, torch, numpy as np, os
from PIL import Image
from torchvision import transforms
from models.embedding_model import EmbeddingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "..", "models", "embedding.pth")
INDEX = os.path.join(BASE, "..", "faiss", "index.bin")
PATHS = os.path.join(BASE, "..", "faiss", "paths.npy")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()

index = faiss.read_index(INDEX)
paths = np.load(PATHS)

def find_similar(image_path, k=6):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)

    D, I = index.search(emb, k)
    return [paths[i] for i in I[0]]
