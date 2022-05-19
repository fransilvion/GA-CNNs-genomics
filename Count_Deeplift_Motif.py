import subprocess
import pandas as pd
import os
from tqdm import tqdm
import sys
import shutil
import re

run_name = sys.argv[1]

run_name = 'Run_28_11_F_10In_100It'
PDBX = os.path.expanduser('~/Dropbox/UBC/')

run_name = PDBX + 'data/Runs_GA/'+ run_name

motif_path = os.path.expanduser(PDBX+'data/deeplift_motifs.meme')

#%%
path_fimo = os.path.expanduser('~/meme/bin/fimo')
if os.path.isdir('{}/Fimo'.format(run_name)) :
    shutil.rmtree('{}/Fimo'.format(run_name), ignore_errors=True)
os.mkdir('{}/Fimo'.format(run_name))

#%% Apply motif count by filter for the population
motif_counts = {}
for f in tqdm(os.listdir('{}/Population/'.format(run_name))) :
    n_sequences = int(subprocess.check_output("grep '>' '{}/Population/{}' | wc -l".format(run_name,f),shell=True).strip())
    fimo_out_path = '{}/Fimo/fimo_{}.tsv'.format(run_name, f.replace('.fa',''))
    cmd = '{} --text --verbosity 1 {} {}/Population/{} > {}'.format(path_fimo, motif_path, run_name, f, fimo_out_path)
    subprocess.call(cmd, shell=True)
    df = pd.read_csv(fimo_out_path, sep="\t")
    motif_counts[f.replace('.fa','')] = (df.groupby('# motif_id')['sequence_name'].nunique()/n_sequences).to_dict()
cdf = pd.DataFrame(motif_counts).fillna(0)
#%%

motif_counts = {}
for f in tqdm(os.listdir('{}/Population/'.format(run_name))) :
    n_sequences = int(subprocess.check_output("grep '>' '{}/Population/{}' | wc -l".format(run_name,f),shell=True).strip())
    fimo_out_path = '{}/Fimo/fimo_{}.tsv'.format(run_name, f.replace('.fa',''))
    df = pd.read_csv(fimo_out_path, sep="\t")
    motif_counts[f.replace('.fa','')] = (df.groupby('# motif_id')['sequence_name'].nunique()/n_sequences).to_dict()
cdf = pd.DataFrame(motif_counts).fillna(0)

#%%
cdf



ttable = pd.read_csv(PDBX+'data/deeplift_summary_sasha.csv')


cdf_n = pd.merge(cdf, ttable[['Query_ID','Target_ID']], how='left', left_index=True, right_on = 'Query_ID')
cdf_n['Target_ID'].nunique()


cdf_n

#%% Load list 99 motifs
with open(motif_path) as f:
    list_task_deeplift = re.findall(r'\b(\w*task\w*)\b', f.read())
#%% Load and preprocess excel file
table  = pd.read_excel(PDBX+'data/tableS1A.xlsx',index_col=1)
a = table.iloc[0]
table = table[1:]
table.columns = a
tdf = table.loc[list_motifs_99,['Best overall TF match','Motif consensus Sequence']]




#%% Merge
fdf = pd.merge(cdf_n,tdf,left_index=True,right_index=True)
fdf['Best overall TF match'] = fdf['Best overall TF match'].fillna('MOTIF NOT KNOWN')
fdf['Best overall TF match'] = fdf.apply(lambda l : l['Best overall TF match'].split('/')[0],axis=1)
fdf['Best overall TF match'] = fdf.apply(lambda l : l['Best overall TF match'].split(' and ')[0],axis=1)



fdf = fdf.loc[sorted(fdf.index,key = lambda x : int(x.replace('filter','')))]
'filter93' in list(fdf.index)
fdf.to_csv('{}/global_count_motifs_deeplift.csv'.format(run_name))
