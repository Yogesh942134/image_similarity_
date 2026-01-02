import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def evaluate_recall(model, dataset, device, k=(1,5), batch=256):
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    embeddings, labels = [], []

    for imgs, lbls in tqdm(loader, desc="Extracting embeddings"):
        imgs = imgs.to(device)
        emb = model(imgs).cpu()
        embeddings.append(emb)
        labels.append(lbls)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    N = embeddings.size(0)
    recalls = {kk:0 for kk in k}

    for i in tqdm(range(0, N, batch), desc="Recall@K"):
        q = embeddings[i:i+batch]
        q_labels = labels[i:i+batch]

        d = torch.cdist(q, embeddings)   # (B,N)
        d[:, i:i+batch] = 1e9            # ignore self

        for kk in k:
            nn = torch.topk(d, kk, largest=False)[1]
            match = (labels[nn] == q_labels.unsqueeze(1)).any(1).sum()
            recalls[kk] += match.item()

    for kk in recalls:
        recalls[kk] /= N

    return recalls
