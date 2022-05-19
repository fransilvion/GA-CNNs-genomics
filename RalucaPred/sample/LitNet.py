import pytorch_lightning as pl
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch
import pandas as pd
from functools import reduce
from utils.sequences_utils import seq_letters_num, seq_one_hot
import torch.optim as optim


class LitNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        num_filters_c1 = 100
        out_fc = 200

        self.cpa1 = nn.ConstantPad1d(15 // 2, 0.25)
        self.c1 = nn.Conv1d(4, num_filters_c1, 15)
        self.llr = nn.LeakyReLU(0.1)
        self.mp1 = nn.AdaptiveMaxPool1d(1)
        self.ap1 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(num_filters_c1 * 2, out_fc)
        self.fc2 = nn.Linear(out_fc, out_fc)
        self.fcf = nn.Linear(out_fc, len(self.hparams.column_names))
        self.fa = nn.Sigmoid()

    def forward(self, x):
        x = self.cpa1(x)
        x = self.llr(self.c1(x))
        x1 = self.mp1(x)
        x2 = self.ap1(x)
        x = torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)  # concat along last dimension
        x = self.llr(self.fc1(x))
        x = self.llr(self.fc2(x))
        x = self.fa(self.fcf(x))
        return x

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x, y = batch
        o = self(x)
        loss = F.mse_loss(o, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x, y = batch
        o = self(x)
        loss = F.mse_loss(o, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': logs}

    def test_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x, y = batch
        o = self(x)
        loss = F.mse_loss(o, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    @staticmethod
    def excel_ra_to_numpy(path_excel: str, columns: List[str]) -> Tuple[np.array, np.array, np.array]:
        """ Open excel file with given format ( from the dataset ), normalise the activations
        :param path_excel: path of the excel file to return
        :param columns: list of the names of the columns to load
        :return np.array : Sequences, np.array : Normalised activations
        """
        data = pd.read_excel(path_excel)
        data = data[data['ID'].str.contains('Bound')]
        y = data.loc[:, columns].to_numpy()
        x = data['Sequence'].to_numpy()
        idxs = np.array(range(len(y)))
        xoh = np.array([seq_one_hot(seq_letters_num(np.array(list(s))).astype(int)).numpy().T for s in x])
        return idxs, xoh, y

    def prepare_data(self) -> None:
        print(f'Loaded Data From {self.hparams.data_path} : Columns : {self.hparams.column_names}')

        idxs, xoh, y = self.excel_ra_to_numpy(self.hparams.data_path, self.hparams.column_names)
        if self.hparams.__contains__('data_repartition'):
            print('Using Existing Repartition')
            x_train = xoh[self.hparams.data_repartition['train']]
            y_train = y[self.hparams.data_repartition['train']]

            x_val = xoh[self.hparams.data_repartition['val']]
            y_val = y[self.hparams.data_repartition['val']]

            x_test = xoh[self.hparams.data_repartition['test']]
            y_test = y[self.hparams.data_repartition['test']]

        else:
            x_train, x_temp, y_train, y_temp, idxs_train, idxs_temp = train_test_split(xoh, y,
                                                                                       idxs, test_size=0.2)
            x_val, x_test, y_val, y_test, idxs_val, idxs_test = train_test_split(x_temp, y_temp,
                                                                                 idxs_temp, test_size=0.5)
            self.hparams.data_repartition = {'train': idxs_train, 'val': idxs_val, 'test': idxs_test}

        y_min = y_train.min(axis=0)
        y_max = y_train.max(axis=0)

        self.act_normalize = lambda s: (s - y_min) / (y_max - y_min)

        [y_train, y_val, y_test] = list(map(self.act_normalize, [y_train, y_val, y_test]))

        self.datasets = {
            'train': TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
            'val': TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()),
            'test': TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()),
        }
        print('Repartion : \n\t{} : {}\t{} : {}\t{} : {}'\
              .format(*reduce(lambda a, b: a + (b[0], len(b[1])), self.datasets.items(), ())))

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(dataset=self.datasets['train'],
                                           batch_size=self.hparams.batch_size,
                                           num_workers=10,
                                           shuffle=True,
                                           drop_last=False)

    def val_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(dataset=self.datasets['val'],
                                           batch_size=self.hparams.batch_size,
                                           num_workers=10,
                                           shuffle=False,
                                           drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(dataset=self.datasets['test'],
                                           batch_size=self.hparams.batch_size,
                                           # num_workers=10,
                                           shuffle=False,
                                           drop_last=False)

    def configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        return optim.Adam(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)
