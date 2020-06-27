import torch
from torch.utils.data import Dataset


class NlpDataset(Dataset):
    def __init__(self, data: tuple):
        sequences, mask, labels = data
        self.x = torch.Tensor(sequences).long()
        self.y = torch.Tensor(labels).long()
        self.mask = torch.ByteTensor(mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.mask[item]
