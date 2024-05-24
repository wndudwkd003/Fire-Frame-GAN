import os
from torch.utils.data import Dataset
from PIL import Image


class FireFlameDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        for root, _, files in os.walk(root):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image