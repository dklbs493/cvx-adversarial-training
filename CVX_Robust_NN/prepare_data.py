import torch
from torch.utils.data import Dataset

class PrepareData(Dataset):
    def __init__(self, X, y, transforms=None, numpy=False):
        if not torch.is_tensor(X) and numpy == False:
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        self.transforms = transforms
        if not torch.is_tensor(y) and numpy == False:
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        out = self.X[idx]
        if self.transforms:
          out = self.transforms(self.X[idx])
        return out, self.y[idx]


class PrepareData3D(Dataset):
    def __init__(self, X, y, z):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]