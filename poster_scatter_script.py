import pandas as pd
import plotly.graph_objects as go
from ipywidgets import interact_manual, widgets
import torch
import sys
import os
sys.path.append(os.path.expanduser('~/Desktop/Basset_E/'))
from sample.utils.load_model import load_model_E
import numpy as np
from Interpretation.utils import *
import json
import seaborn as sns

table  = pd.read_excel(PDBX+'data/tableS1A.xlsx',index_col=1)
a = table.iloc[0]
table = table[1:]
table.columns = a


run = '/home/emeunier/Dropbox/UBC/data/Runs_GA/Random_Population'
run2 = '/home/emeunier/Dropbox/UBC/data/Runs_GA/Run_28_11_F_10In_100It'

# Sasha motifs generated count
generated_smotifs = pd.read_csv('{}/global_count_motifs_sasha.csv'.format(run),index_col=0)
generated_smotifs_2 = pd.read_csv('{}/global_count_motifs_sasha.csv'.format(run2),index_col=0)

# Load Model ( please use the same as the one for motif generation)
t = torch.load(os.path.expanduser('~/Dropbox/UBC/Models/Basset_2019-11-04_19_23_35/model_loss_0.68.tar'))
m, _ = load_model_E()
m.load_state_dict(t['model_state_dict'])
m.eval();
activation = lambda x : 8*torch.sigmoid(x)+1
# Load Data
x = np.load(os.path.expanduser(os.path.expanduser('~/Dropbox/UBC/data/perm_one_hot_seqs.npy'))) #load permuted sequences
x = x.astype(np.float32)

be_importance = pd.read_csv(os.path.expanduser('~/Dropbox/UBC/data/Activation_BassetE_FilterS.csv'),index_col=0).drop(columns=['Best overall TF match','Motif consensus Sequence'])
comparaison_be_2 = pd.merge(be_importance,
                       generated_smotifs_2,
                       left_index=True, right_index=True, suffixes=('_be','_generated'), how='left')
comparaison_be_2 = comparaison_be_2.fillna(0)
aga = pd.merge(comparaison_be_2,table[['Reproducibility']], left_index=True, right_index=True)

sns.scatterplot(x='B_be', y='B_generated', hue='Reproducibility', palette='rainbow', s=20, linewidth=0, data=aga)
