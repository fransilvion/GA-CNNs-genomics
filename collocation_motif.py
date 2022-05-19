# The goal of this notebook is to evaluate if 2 motifs are often found in the same sequence
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import re
import chart_studio.plotly as py
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

PDBX = os.path.expanduser('~/Dropbox/UBC/')

def read_list_motif(list_motif_path, excel_path) :
    """
    Read the list of selected motif and the excel description of the motifs to return a dataframe containing both
    :param list_motif_path : path of the meme file containing the motif we selected
    :param excel_path : path of the excel file with the information on the motif_path
    :return tdf : DataFrame with the name of the motif in the meme file as index and then the real name of the motif and finally the consensus
    """
    motif_path = os.path.expanduser(list_motif_path)
    with open(motif_path) as f:
        list_motifs_99 = re.findall(r'\b(\w*filter\w*)\b', f.read())
    table  = pd.read_excel(excel_path,index_col=1)
    a = table.iloc[0]
    table = table[1:]
    table.columns = a
    table.fillna('Not Known', inplace=True)
    tdf = table.loc[list_motifs_99,['Best overall TF match','Motif consensus Sequence']]
    return tdf




def compute_distance(st1, e1, st2, e2) : # est ce qu'on s'en fiche de l'ordre ( a verifier)
    """
    Compute distance / overlap between 2 sequences, positive is distance, negative is overlap
    :param st* : start sequence *
    :param e* : end sequence *

    TEST :
            Questions :
                test_unit = [(1,2,3,4),(1,3,2,4),(1,4,2,3),(3,4,1,2),(2,3,1,4),(2,4,1,3)]
                for c in test_unit :
                    print(compute_distance(*c))
            Expected Output :
                1, -1, -1, 1, -1, -1
    """
    if st1 < st2 :
        if e2>e1 :
            return st2-e1
        else :
            return st2-e2
    else :
        if e1>e2 :
            return st1-e2
        else :
            return st1-e1

def build_distance_matrix(dfimo, list_motif = None) :
    """
    Take a dataframe build from a Fimo file containing the motif scanned in a serie of sequences and their position and build a distance matrix
    of the shape sequence*n_motif*n_motif that indicate for each sequence the distance between each pair of Motifs, use the function compute distance
    so see this function for additional details on distance computation.
    :param dfimo : Dataframe extracted from fimo file
    :param list_motif : list of motif to add to the matrix, if not given will be infer from the fimo file
    :return distances : matrix with the distance of shape sequence*n_motif*n_motif, sequences : list sequences we have in axis 0 in order, list_motif : list motif we have in order
    """
    sequences = set(dfimo['sequence_name'])
    if list_motif is None :
        list_motif =  list(set(dfimo['# motif_id']))
    distances = np.empty((len(sequences),len(list_motif),len(list_motif)))
    distances[:] = np.nan

    for i, s in tqdm(enumerate(sequences), total=len(sequences)) :
        t = dfimo[dfimo['sequence_name'] == s].drop_duplicates(subset='# motif_id')
        for a in t.iterrows() :
            for b in t.iterrows() :
                distances[i, list_motif.index(a[1]['# motif_id']), list_motif.index(b[1]['# motif_id'])] = compute_distance(a[1]['start'],a[1]['stop'], b[1]['start'], b[1]['stop'])
    return distances, sequences, list_motif

def load_distance_matrix(distances_f, dfimo, list_motif=None) :
    """
    Load te distance matrix from te file.
    :param distances_f : numpy file containing te distance matrix
    :param dfimo : Dataframe extracted from fimo file
    :param list_motif : list of motif to add to the matrix, if not given will be infer from the fimo file
    :return distances : matrix with the distance of shape sequence*n_motif*n_motif, sequences : list sequences we have in axis 0 in order, list_motif : list motif we have in order
    """
    sequences = set(dfimo['sequence_name'])
    if list_motif is None :
        list_motif =  list(set(dfimo['# motif_id']))
    distances = np.load(distances_f)
    return distances, sequences, list_motif

def map_motif_colocation(distances, tdf, th_over, th_min_seq, normalisation, list_motif) :
    """
        Take the distance matrix and create a Heatmap representing the colocation of the motifs, the colocation being the number of unique sequences in the dataset containing both motifs
        :param distances : matrix with the distance of shape sequence*n_motif*n_motif
        :param tdf : Dataframe with information over the motif for labelling
        :param th_over : overlap threshold , if the motifs are overlaping over this threshold this collocation is not counted. -1 if no filter
        :param th_min_seq : minimum of unique sequences that should contain the motif for it to be added to the map, allows to have too much motif on the map. False if no filter
        :param normalisation : if True we divide the count by the number of sequence containing one of the 2 motifs ( IoU )
        :return map_collocation : Dataframe dimension n_motif * n_motif with the cout for each pair, count_motifs : total count for each motif
    """
    names = np.array([a+' - '+tdf.loc[a, 'Best overall TF match'].split('/')[0].strip() for a in np.array(list_motif)]) # extract motifs
    count_motifs = np.sum((np.sum(np.isnan(distances) == False,axis=1) > 0), axis=0) # count total number of unique sequences containing each motif
    if th_over is not False :
        #print('Threshold Overlap : {}'.format(th_over))
        dt = (distances>=th_over).sum(axis=0) # comput map
    else :  dt = distances.sum(axis=0)
    if normalisation :
        countor = (((np.zeros((len(count_motifs),len(count_motifs)))+np.array(count_motifs)).T)+np.array(count_motifs)) - dt # (f1 U f2 + f1 N f2 - f1 N f2)
        dt = dt/countor
    dt = dt[count_motifs > th_min_seq][:,count_motifs>th_min_seq] # filter the rows
    names = names[count_motifs > th_min_seq]
    dftf= pd.DataFrame(dt,columns=names, index=names)
    return dftf, count_motifs
    #sns.clustermap(dftf)
    #go.Figure(go.Heatmap(dftf))

def scatter_collocation(n_sequences, count_motifs, map_distances, name_save) :
    """
        Build a scatter that compare for each pair of motif in the dataset the collocation count with the expected count, the collocation
        is extracted from sequences and the expected count suppose that each presence of motif is independent from the other
        it's an independence test, if the presence of the motif is interdependent we should not see a line on the graph.
        :param n_sequences : total number of sequences
        :param count_motifs : total number of individual sequence where the motif appear across the data
        :param map_distances :  motif collocation counted on the data for each pair
        :param name_save : path where to save the image
    """
    # We compute a matrix with the expected count for each pair if the presence of the motif is independent : E(M1, M2) = P(M1) * P(M2) * Total sequences =() N_M1/Total_sequences * N_M2* Total_sequences)* Total_sequences
    expe = np.dot(count_motifs.reshape(1,-1).T,count_motifs.reshape(1,-1))/n_sequences
    np.fill_diagonal(expe,0)
    plt.scatter(x=expe.flatten(), y = map_distances.to_numpy().flatten())
    plt.xlabel('number of expected pairs')
    plt.ylabel('number of observed pairs')
    plt.savefig(name_save)
    plt.clf()


#%% Build the dataset of OCR / Motifs
distances_file = PDBX+'/data/Runs_GA/Run_28_11_F_10In_100It/distances_motif_B.npy'
tdf = read_list_motif(PDBX+'data/filter_motifs_pwm99.meme', PDBX+'data/tableS1A.xlsx')
dfimo = pd.read_csv(PDBX+'data/Runs_GA/Run_28_11_F_10In_100It/Fimo/fimo_B.tsv', sep='\t')
if distances_file :
    distances, sequences, list_motif = load_distance_matrix(distances_file, dfimo, list(tdf.index))
else :
    distances, sequences, list_motif = build_distance_matrix(dfimo, list(tdf.index))

for th_over in range(-25,50,5) :
    th_min_seq = -1
    map_distances, count_motifs = map_motif_colocation(distances, tdf, th_over, th_min_seq, False, list_motif)
    name_path = 'maps/collocation_motif_B_count_thover({})_thminseq({})'.format(th_over, th_min_seq)
    map_distances.to_csv(name_path+'.csv')
    scatter_collocation(distances.shape[0], count_motifs, map_distances, name_path+'.png')


#%% Naive extraction of co-occurence
expe = np.dot(count_motifs.reshape(1,-1).T,count_motifs.reshape(1,-1))/distances.shape[0]
np.fill_diagonal(expe,0)
fig = go.Figure(go.Scatter(x=expe.flatten(), y=map_distances.to_numpy().flatten()))
fig.write_image('test.png')              
