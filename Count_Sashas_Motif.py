'''
Compare the number of motif found in the database (.meme) file in the different populations (.fa) of one run
Foler Structure to follow :
- Run Name :
    - Population :
        - pop1.fa ( exemple B-Cells)
        - pop2.fa
        ...
        - popl.fa

The output will be under the same folder and will consist of the files from FIMO and a csv summarizing the differnt counts.
'''


import subprocess
import pandas as pd
import os
from tqdm import tqdm
import sys
import shutil
import re
import argparse


def count_sasha_motif(run_name, database_path) :

    path_fimo = os.path.expanduser('~/meme/bin/fimo')
    if os.path.isdir('{}/Fimo'.format(run_name)) :
        shutil.rmtree('{}/Fimo'.format(run_name), ignore_errors=True)

    #%% Apply motif count by filter for the population
    motif_counts = {}
    for f in tqdm(os.listdir('{}/Population/'.format(run_name))) :
        n_sequences = int(subprocess.check_output("grep '>' '{}/Population/{}' | wc -l".format(run_name,f),shell=True).strip())
        fimo_out_path = '{}/Fimo/fimo_{}'.format(run_name, f.replace('.fa',''))
        if not os.path.isdir(fimo_out_path):
            os.makedirs(fimo_out_path)
        cmd = 'fimo -qv-thresh --thresh 0.05 -verbosity 1 -oc {} {} {}/Population/{}'.format(fimo_out_path, database_path, run_name, f)
        print(cmd)
        subprocess.call(cmd, shell=True)
        df = pd.read_csv(fimo_out_path+'/fimo.txt', sep="\t")
        motif_counts[f.replace('.fa','')] = (df.groupby('# motif_id')['sequence_name'].nunique()/n_sequences).to_dict()
    cdf = pd.DataFrame(motif_counts).fillna(0)

    #%% Load list 99 motifs
    with open(database_path) as f:
        list_motifs_99 = re.findall(r'\b(\w*filter\w*)\b', f.read())

    #%% Load and preprocess excel file
    table  = pd.read_excel(PDBX+'data/sasha_ATAC/tableS1A.xlsx',index_col=1)
    a = table.iloc[0]
    table = table[1:]
    table.columns = a
    tdf = table.loc[list_motifs_99,['Best overall TF match','Motif consensus Sequence']]

    #%% Merge
    fdf = pd.merge(cdf,tdf,left_index=True,right_index=True, how='right')
    fdf = fdf.fillna(0)
    fdf['Best overall TF match'] = fdf['Best overall TF match'].replace(0,'MOTIF NOT KNOWN')
    fdf['Best overall TF match'] = fdf.apply(lambda l : l['Best overall TF match'].split('/')[0],axis=1)
    fdf['Best overall TF match'] = fdf.apply(lambda l : l['Best overall TF match'].split(' and ')[0],axis=1)

    fdf = fdf.loc[sorted(fdf.index,key = lambda x : int(x.replace('filter','')))]
    fdf.to_csv('{}/global_count_motifs_sasha.csv'.format(run_name))


if __name__=='__main__':
    PDBX = os.environ['PDBX_UBC']
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', help='Path of the Run Folder to Use to use')
    parser.add_argument('-d', '--database', help='Path of the Database (.meme file) to use',
                                            default=PDBX+'data/sasha_ATAC/filter_motifs_pwm99.meme')
    args = parser.parse_args()
    count_sasha_motif(args.run_name, args.database)
