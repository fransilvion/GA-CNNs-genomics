from sample.LitNet import LitNet
import pytorch_lightning as pl
from glob import glob
from argparse import ArgumentParser
import os
import warnings
import tensorflow as tf
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#  Define Parameters
parser = ArgumentParser()

parser.add_argument("--data_path", help="path of the activation file (.npy)", default=None, type=str)

parser.add_argument("--batch_size", help="Batch Size", default=200, type=int)
parser.add_argument("--weight_decay", help="Value for weight decay with adam", default=0, type=float)
parser.add_argument("--learning_rate", help="Value for learning rate with adam", default=3e-4, type=float)

parser = pl.Trainer.add_argparse_args(parser)
hparams, ukn = parser.parse_known_args()

hparams.data_path = os.environ['PDBX_UBC'] + '/data/GA/Raluca/Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx'  # Predictor
hparams.column_names = ['E2f4_500nM', 'E2f4_800nM']

# hparams.data_path = os.environ['PDBX_UBC'] + '/data/GA/Raluca/Combined_Max_Myc_Mad_Mad_r_log.xlsx'  # Predictor
# hparams.data_path = os.environ['PDBX_UBC'] + '/data/GA/Raluca/Filtered_Max.xlsx'
# hparams.column_names = ['Max']


model = LitNet(hparams)
trainer = pl.Trainer(log_gpu_memory='all',
                     max_epochs=300,
                     early_stop_callback=True)
trainer.fit(model)
trainer.test(model)
'''
preds = []
gt = []
for x, y in model.test_dataloader():
    preds.append(model(x))
    gt.append(y)
preds = torch.cat(preds).flatten().detach()
gt = torch.cat(gt).flatten().detach()
plt.scatter(preds, gt, label='Test data')

print(spearmanr(preds, gt))
plt.show()

fx, fa = model.excel_ra_to_numpy(os.environ['PDBX_UBC'] + '/data/GA/Raluca/Filtered_Max.xlsx')
fa = model.act_normalize(fa)
fx = torch.tensor(fx)
fa = torch.tensor(fa)
fap = model(fx).flatten().detach()
plt.scatter(fap, fa, label='Test data')
plt.legend()
plt.show()
'''
