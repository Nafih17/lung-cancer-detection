import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_samples = len(image_paths)

    def __getitem__(self, index):
        img_path, mask_path = self.image_paths[index], self.mask_paths[index]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        return torch.tensor(image), torch.tensor(mask)

    def __len__(self):
        return self.num_samples
