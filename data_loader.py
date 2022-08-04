import os
import torch
from PIL import Image
import torch.utils.data

class VimeoDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, stage, transform=None):
        self.data_dir = os.path.join(data_path, stage)
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx], 'im1.png')
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

class KodakDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img