from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class CustomDataSet(Dataset):
    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 transform: bool = True,
                 ):
        self.images = images.copy()
        self.labels = labels.copy()
        if transform:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(0.1307, 0.3081)])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        image = self.transforms(image)
        return image.to(torch.float32), label





