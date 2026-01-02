from torch.utils.data import Dataset
from PIL import Image
import os

class TripletMyData(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for label, cls in enumerate(os.listdir(root)):
            cls_path = os.path.join(root, cls)
            for img in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img), label))

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)
