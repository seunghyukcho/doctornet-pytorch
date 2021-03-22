import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

class LabelMeDataset(Dataset):
    def __init__(self, root_dir, is_train=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.is_train = is_train
        self.labels = np.loadtxt(self.root_dir / 'labels.txt', dtype=np.float32)
        self.transforms = transforms.Compose([transforms.Resize(299), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        with open(self.root_dir / 'filenames.txt', 'r') as f:
            self.image_paths = f.read().splitlines()
        if self.is_train:
            self.annotations = np.loadtxt(self.root_dir / 'annotations.txt', dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        x = Image.open(self.image_dir / self.image_paths[idx])
        x = self.transforms(x)
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y

