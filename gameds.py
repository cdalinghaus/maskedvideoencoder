from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os


class GameDS(Dataset):
    
    def __init__(self, path, train=True, sequence_length=10):
        if train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "val")

        self.sequences = os.listdir(path)

        self.sequences = [x for x in self.sequences if not "." in x]

        self.files = []
        for sequence in self.sequences:
            local_path = os.path.join(path, sequence)
            local_frames = sorted(os.listdir(local_path))
            self.files += [local_path + "/" + x for x in local_frames]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files) - self.sequence_length
    
    def __getitem__(self, idx):
        files = self.files[idx:idx+self.sequence_length]
        
        images = [Image.open(f) for f in files]
        tensors = [torch.tensor(np.array(im)) for im in images]
        stacked = torch.stack(tensors)
        result = stacked
        result = result.moveaxis(3, 1)
        return result, torch.Tensor([0])