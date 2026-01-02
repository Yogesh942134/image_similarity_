import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from training.triplet_mydata import TripletMyData
from training.loss import BatchHardTripletLoss
from models.embedding_model import EmbeddingNet
from tqdm import tqdm
from training.sampler import BalancedBatchSampler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = TripletMyData("data/mydata", transform)
    labels = [l for _,l in dataset.samples]

    sampler = BalancedBatchSampler(labels, 4, 16)  # 64 batch
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    model = EmbeddingNet().to(device)
    criterion = BatchHardTripletLoss(0.2).to(device)

    for p in model.backbone.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), 3e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Head warmup
    for epoch in range(5):
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print("Warmup epoch", epoch+1, "done")

    # Full training
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    for epoch in range(25):
        total = 0
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss {total/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/embedding.pth")
    print("Training complete")

    from training.recall_eval import evaluate_recall

    print("Evaluating Recall@K ...")
    recall = evaluate_recall(model, dataset, device)
    print("Recall:", recall)

if __name__ == "__main__":
    main()

