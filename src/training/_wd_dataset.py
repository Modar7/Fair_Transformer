import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset

from src.wdtypes import * 


class DatasetObject(Dataset):
    def __init__(
        self,
        X_tab: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Any] = None,
    ):
        super(DatasetObject, self).__init__()
        self.X_tab = X_tab
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx: int):  
        X = Bunch()
        if self.X_tab is not None:
            X.deeptabular = self.X_tab[idx]

        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        if self.X_tab is not None:
            return len(self.X_tab)
