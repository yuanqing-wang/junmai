import torch
import lightning as pl
import os
import numpy as np
from pathlib import Path
import requests
from torch.utils.data import TensorDataset, DataLoader
CACHE_DIR = os.path.join(Path(__file__).parent, ".cache/")
print(CACHE_DIR)

class CCSD(pl.LightningDataModule):
    def __init__(
            self, 
            name: str,
            batch_size: int = 32, 
            num_workers: int = 1, 
            num_train: int = 950,
            num_val: int = 50,
            normalize: bool = True
        ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.normalize = normalize

    def setup(self, stage=None):
        url = f"http://www.quantum-machine.org/gdml/data/npz/{self.name}_ccsd_t.zip"
        self.train_path = os.path.join(CACHE_DIR, f"{self.name}_ccsd_t-train.npz")
        self.test_path = os.path.join(CACHE_DIR, f"{self.name}_ccsd_t-test.npz")
        if not os.path.exists(self.train_path):
            os.makedirs(CACHE_DIR, exist_ok=True)
            r = requests.get(url, allow_redirects=True)
            open(self.train_path, 'wb').write(r.content)
            # unzip the file
            import zipfile
            with zipfile.ZipFile(self.train_path, 'r') as zip_ref:
                zip_ref.extractall(CACHE_DIR)

        data_train = np.load(self.train_path)
        R, E, F = data_train['R'], data_train['E'], data_train['F']
        Z = data_train['z'][None, :].repeat(R.shape[0], 0)
        R, E, F, Z = map(
            lambda x: torch.tensor(x, dtype=torch.float32),
            (R, E, F, Z)
        )
        Z = torch.nn.functional.one_hot(Z.long()).float()
        idxs = np.random.permutation(R.shape[0])
        self.R_tr = R[idxs[:self.num_train]]
        self.E_tr = E[idxs[:self.num_train]]
        self.F_tr = F[idxs[:self.num_train]]
        self.Z_tr = Z[idxs[:self.num_train]]
        # self.R_vl = R[idxs[self.num_train:self.num_train+self.num_val]]
        # self.E_vl = E[idxs[self.num_train:self.num_train+self.num_val]]
        # self.F_vl = F[idxs[self.num_train:self.num_train+self.num_val]]
        # self.Z_vl = Z[idxs[self.num_train:self.num_train+self.num_val]]
        self.R_vl = self.R_tr
        self.E_vl = self.E_tr
        self.F_vl = self.F_tr
        self.Z_vl = self.Z_tr

        data_test = np.load(self.test_path)
        self.R_te, self.E_te, self.F_te = data_test['R'], data_test['E'], data_test['F']
        self.Z_te = data_test['z'][None, :].repeat(self.R_te.shape[0], 0)
        self.R_te, self.E_te, self.F_te, self.Z_te = map(
            lambda x: torch.tensor(x, dtype=torch.float32),
            (self.R_te, self.E_te, self.F_te, self.Z_te)
        )
        self.Z_te = torch.nn.functional.one_hot(self.Z_te.long()).float()

        self.E_MEAN, self.E_STD = self.E_tr.mean(), self.E_tr.std()

        if self.normalize:
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