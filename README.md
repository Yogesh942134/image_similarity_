### 🔍 AI Fashion Recommendation Search Engine

An end-to-end deep learning powered image similarity system that allows users to upload any fashion image and instantly find visually similar products from a large-scale Google Drive hosted dataset.

Built using Triplet Network, FAISS vector search, GPU acceleration, and Streamlit deployment.     

### 🚀 Live Demo

👉 https://imagesimilarity-21.streamlit.app/

## 🎥 Project Demo

[▶ Watch Full Demo](https://drive.google.com/file/d/1ZlTBxIH45b0xpTjNCQnMQY-vLpDTKT1A/view)


### 🧠 Key Features

• Deep learning image embeddings using ResNet50 Triplet Network
• Batch Hard mining for high accuracy similarity learning
• Recall@K evaluation pipeline
• FAISS GPU accelerated vector indexing
• Google Drive CDN based massive dataset hosting
• Streamlit based web interface
• Scales to 50K+ fashion images

### 🏗️ System Architecture

                ┌──────────────┐
                │  User Upload │
                └───────┬──────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Streamlit Web App  │
              │   (User Interface) │
              └─────────┬─────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │ Triplet Embedding Model  │
            │     (ResNet50 backbone)  │
            └─────────┬───────────────┘
                      │
                      ▼
            ┌─────────────────────────┐
            │    FAISS Vector Index    │
            │   (Cosine Similarity)    │
            └─────────┬───────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │ Google Drive CDN Image Repository   │
     │      (47,000+ Fashion Images)       │
     └────────────────────────────────────┘

    
  ## 📁 Project Architecture

| Folder / File | Description                   |
| ------------- | ----------------------------- |
| app.py        | Streamlit UI                  |
| models/       | Triplet embedding network     |
| training/     | Model training & evaluation   |
| inference/    | FAISS indexing & search       |
| tools/        | Google Drive dataset pipeline |

---

### 🧠 Model Layer

| Path | Purpose |
|-----|--------|
| models/embedding_model.py | ResNet50 based Triplet Network |
| models/embedding.pth | Trained embedding weights |

---

### 🏋️ Training Pipeline

| Path | Purpose |
|-----|--------|
| training/train.py | Triplet network training |
| training/triplet_mydata.py | Dataset loader |
| training/loss.py | Triplet margin loss |
| training/sampler.py | Batch-hard mining |
| training/recall_eval.py | Recall@K evaluation |
| training/run_recall.py | Recall evaluation runner |
| training/data/ | Local dataset |

---

### 🔎 Inference & Search

| Path | Purpose |
|-----|--------|
| inference/build_index_mydata.py | Builds FAISS vector index |
| inference/search.py | Similarity search engine |
| inference/drive_urls.csv | CDN image database |
| inference/faiss/index.bin | FAISS index |
| inference/faiss/paths.npy | Image CDN paths |

---

### ☁️ Google Drive Integration

| Path | Purpose |
|-----|--------|
| tools/drive_to_csv.py | Export Drive dataset to CSV |
| tools/convert_drive_links.py | Convert Drive links to CDN |
| tools/convert_to_cdn.py | Fast CDN link converter |
| tools/rebuild_paths_from_index.py | Repairs FAISS paths |
| tools/client_secrets.example.json | OAuth template |
| tools/settings.yaml | Drive API config |

---

### 🚀 System Overview

| Layer | Function |
|-----|--------|
| Training | Learns image embeddings |
| Inference | Builds vector index |
| FAISS | Performs ultra-fast search |
| Google Drive CDN | Hosts images |
| Streamlit UI | User interface |

### 📦 Installation
- git clone https://github.com/Yogesh942134/image_similarity_.git
- cd image_similarity_
- pip install -r requirements.txt

▶️ Run Locally
- streamlit run app.py

🧪 Recall@K Evaluation
- python training/run_recall.py

⚡ Build FAISS Index
- python inference/build_index_mydata.py

### 📌 Tech Stack

• PyTorch
• FAISS
• Streamlit
• Google Drive API
• ResNet50
• CUDA
