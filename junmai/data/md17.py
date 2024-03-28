import torch
import lightning as pl
import os
import numpy as np
from pathlib import Path
import requests
from torch.utils.data import TensorDataset, DataLoader
CACHE_DIR = os.path.join(Path(__file__).parent, ".cache/")

class MD17(pl.LightningDataModule):
    def __init__(
            self, 
            name: str,
            batch_size: int = 32, 
            num_workers: int = 1, 
            num_train: int = 950,
            num_val: int = 50,
        ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.E_STD = None
        self.E_MEAN = None

    def setup(self, stage=None):
        url = f"http://www.quantum-machine.org/gdml/data/npz/md17_{self.name}.npz"
        self.file_path = os.path.join(CACHE_DIR, f"md17_{self.name}.npz")
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(self.file_path):
            r = requests.get(url, allow_redirects=True)
            open(self.file_path, 'wb').write(r.content)
        data = np.load(self.file_path)
        self.R, self.E, self.F = data['R'], data['E'], data['F']
        self.Z = data['z'][None, :].repeat(self.R.shape[0], 0)
        self.R, self.E, self.F, self.Z = map(
            lambda x: torch.tensor(x, dtype=torch.float32),
            (self.R, self.E, self.F, self.Z)
        )
        idxs = np.random.permutation(self.R.shape[0])
        self.R_tr = self.R[idxs[:self.num_train]]
        self.E_tr = self.E[idxs[:self.num_train]]
        self.F_tr = self.F[idxs[:self.num_train]]
        self.Z_tr = self.Z[idxs[:self.num_train]]
        self.R_vl = self.R[idxs[self.num_train:self.num_train+self.num_val]]
        self.E_vl = self.E[idxs[self.num_train:self.num_train+self.num_val]]
        self.F_vl = self.F[idxs[self.num_train:self.num_train+self.num_val]]
        self.Z_vl = self.Z[idxs[self.num_train:self.num_train+self.num_val]]
        self.R_te = self.R[idxs[self.num_train+self.num_val:]]
        self.E_te = self.E[idxs[self.num_train+self.num_val:]]
        self.F_te = self.F[idxs[self.num_train+self.num_val:]]
        self.Z_te = self.Z[idxs[self.num_train+self.num_val:]]
        self.E_MEAN = self.E_tr.mean()
        self.E_STD = self.E_tr.std()
        self.E_tr = (self.E_tr - self.E_MEAN) / self.E_STD
        self.F_tr = self.F_tr / self.E_STD
        self.ds_tr = TensorDataset(self.R_tr, self.E_tr, self.F_tr, self.Z_tr)
        self.ds_vl = TensorDataset(self.R_vl, self.E_vl, self.F_vl, self.Z_vl)
        self.ds_te = TensorDataset(self.R_te, self.E_te, self.F_te, self.Z_te)

    def train_dataloader(self):
        return DataLoader(
            self.ds_tr, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_vl, 
            batch_size=len(self.ds_vl), 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_te, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )