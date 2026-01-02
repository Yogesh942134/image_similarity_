import streamlit as st
import torch, faiss, numpy as np, os, tempfile, requests
from PIL import Image
from torchvision import transforms
from io import BytesIO
from models.embedding_model import EmbeddingNet

st.set_page_config(page_title="AI Fashion Search", layout="wide")
st.title("🔍 AI Fashion Recommendation System")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "embedding.pth")
FAISS_DIR = os.path.join(BASE, "inference", "faiss")
INDEX = os.path.join(FAISS_DIR, "index.bin")
PATHS = os.path.join(FAISS_DIR, "paths.npy")

if not os.path.exists(INDEX):
    st.error("FAISS index not found. Please run inference/build_index_mydata.py first.")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_system():
    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()
    index = faiss.read_index(INDEX)
    paths = np.load(PATHS)
    return model, index, paths

@st.cache_data(show_spinner=False)
def fetch_image(url):
    r = requests.get(url, timeout=10)
    return Image.open(BytesIO(r.content)).convert("RGB")

model, index, paths = load_system()

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        query_path = tmp.name

    with st.spinner("Finding similar items..."):
        img = Image.open(query_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy().astype("float32")
            faiss.normalize_L2(emb)

        D, I = index.search(emb, 6)
        results = [paths[i] for i in I[0]]

    left, right = st.columns([1,3])

    with left:
        st.subheader("Query Image")
        st.image(query_path, width=220)

    with right:
        st.subheader("Similar Results")
        cols = st.columns(3)
        for i, p in enumerate(results):
            try:
                img = fetch_image(p)
                cols[i % 3].image(img, width=200)
            except:
                pass
