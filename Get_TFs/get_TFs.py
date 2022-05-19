import torch
import os
import gzip
import pandas as pd
import numpy as np
import sys
import os
import itertools
from collections import Counter
from tqdm import tqdm
from threading import Thread
PDBX = os.path.expanduser('~/Dropbox/UBC/')

def load_datas() :
	#JASPAR motif clusters
	#parse all the motif names
	list_of_all_tfs = []
	with open(PDBX+"data/JASPAR2018_CORE_vertebrates_redundant_pfms_jaspar.txt", "r") as f:
		for line in f:
			if line.startswith(">"):
				line = line.split()
				tf = line[-1].strip()

				if tf not in list_of_all_tfs:
					list_of_all_tfs.append(tf.upper())
	tasks = os.listdir('{}/Results'.format(run_name))
	global_count_motifs = pd.DataFrame(index=[t.replace('.fa','') for t in tasks], columns=list_of_all_tfs)
	return global_count_motifs, tasks

def run_task(task, global_count_motifs,pbar) :
	#go to folders with the Jaspar results
	task = task.replace('.fa','')
	files = os.listdir(run_name+"/Results/" + task + ".fa" + "/")
	tfs = {}
	tfs_pos = {}


	#folder with fasta files
	with open(run_name+"/Population/"+ task + ".fa", "r") as f:
		for line in f:
			if line.startswith(">"):
				line = line.split(">")
				tfs[line[-1].strip()] = []
				tfs_pos[line[-1].strip()] = {}


	for fi in files:
		pbar.update(1)
		with open(run_name+"/Results/" + task + ".fa" + "/" + fi,'rt') as f:
			for line in f:
				line = line.split()
				tf = line[3].strip().upper()
				start = int(line[1].strip())
				end = int(line[2].strip())

				new_key_seq = line[0].strip()

				if int(line[-3]) > thr and new_key_seq in tfs.keys():
					tfs[new_key_seq].append(tf)
					tfs_pos[new_key_seq][tf] = np.arange(int(start), int(end)+1)

	#find overlaps
	if not os.path.isdir(run_name+"/Motifs") :
		os.mkdir(run_name+"/Motifs")

	lens_tfs = []
	for b in tfs.keys() :
		lens_tfs.extend([(ki,len(v)) for ki,v in tfs_pos[b].items()])
	lens_tfs =dict(lens_tfs)

	#save all tfs
	c = Counter(list(itertools.chain.from_iterable(tfs.values())))
	with open(run_name+"/Motifs/"+task+"_motif.txt", "w") as f:
		for k,v in  c.most_common():
			f.write( "{} {} {}/{} {}\n".format(k,
											   v,
											   sum([k in t for t in tfs.values()]),
											   len(tfs.keys()),
											   lens_tfs[k]))
			if k in global_count_motifs.columns :
				global_count_motifs.loc[task,k] = sum([k in t for t in tfs.values()])/len(tfs.keys())

#%%
run_name = sys.argv[1]
#run_name = 'Run_Test'
#%cd Get\ TFs
#%%
#significance threshold
thr = 950
global_count_motifs, tasks  = load_datas()
pbar = tqdm(total=len(tasks)*736)

threads = [Thread(target=run_task, args=(task,global_count_motifs,pbar)) for task in tasks]

for t in threads :
	t.start()
## Running Threads
for t in threads :
	t.join()
global_count_motifs.fillna(0, inplace=True)
global_count_motifs.to_csv(run_name+'/global_count_motifs.csv')
