import torch
from training.triplet_mydata import TripletMyData
from models.embedding_model import EmbeddingNet
from training.recall_eval import evaluate_recall
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = TripletMyData("data/mydata", transform)

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load("models/embedding.pth", map_location=device))
model.eval()

print("Running Recall@K evaluation...")
recall = evaluate_recall(model, dataset, device)
print("Recall:", recall)
