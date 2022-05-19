import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pathlib import Path

#load tomtom results
tomtom = pd.DataFrame()
scores = []
names = []
for i in range(81):
    filename = "./output_task" + str(i) + "/tomtom_out/tomtom.tsv"
    if Path(filename).is_file():
        df = pd.read_csv(filename, sep='\t')
        df = df[:-3]
        tomtom = pd.concat([tomtom, df])

        tmp = np.load("./output_task" + str(i) + "/scores.npy")
        scores.append(tmp)

        with open("./output_task" + str(i) + "/fwd_motifs.meme") as fp:
            line = fp.readline()
            while line:
                if(line[0:5]=="MOTIF"): 
                    names.append(line.replace('MOTIF ','').strip('\n'))
                line = fp.readline() 
        fp.close()

scores = np.vstack(scores)
names = np.vstack(names)
scores = pd.DataFrame(np.mean(scores, axis=(1,2)), index=names.squeeze(), columns=['Ave_Score'])

df = tomtom.sort_values(by='q-value').drop_duplicates(subset='Query_ID', keep='first')

df = pd.merge(scores, df, how='left', left_index=True, right_on='Query_ID')
df['log_qval'] = -np.log(df['q-value'])
df['log_qval'] = df['log_qval'].fillna(0)

#load basset summary
summary = pd.read_csv("motif_summary2.csv") 
summary = summary[['ID', 'Filter ID', 'Influence', 'Reproducibility', 'Best overall TF match']]
summary.columns=['Number', 'Target_ID', 'Influence', 'Reproducibility', 'TF_Name']
summary['Reproducibility'] = summary['Reproducibility']+1

df1 = df.merge(summary, how='left', on='Target_ID')
df1['logInfl'] = np.log(df1['Influence'])
df1['AI-TAC/DeepLift Significant Match'] = df1['q-value']<0.01

df2 = df.merge(summary, how='right', on='Target_ID')
df2 = df2.sort_values(by='q-value').drop_duplicates(subset='Target_ID', keep='first')
df2['log_qval'] = df2['log_qval'].fillna(0) 
df2['logInfl'] = np.log(df2['Influence'])
df2['Filter_Name'] = df2['Number'].astype(str) + " " + df2['TF_Name']

if True:

    sns.set_style(style='white')
    plt.figure(figsize=(11, 6))
    sns.scatterplot(x='Ave_Score', y='log_qval', hue='Reproducibility', palette='rainbow', s=20, linewidth=0, data=df1)
    plt.axhline(y=3.0)
    plt.xlabel("Average DeepLIFT Importance Score", fontsize=15)
    plt.ylabel("Negative Log q-value of Best Match", fontsize=15)

    plt.savefig("./figures/FigS3B_Old.pdf")
    plt.close()


    sns.set_style(style='white')
    plt.figure(figsize=(11, 6))
    sns.scatterplot(x='Ave_Score', y='logInfl', hue='AI-TAC/DeepLift Significant Match', palette='RdBu_r', s=20, linewidth=0, data=df1)
    plt.xlabel("Average DeepLIFT Importance Score", fontsize=15)
    plt.ylabel("Log of Influence of Best Match", fontsize=15)

    plt.savefig("./figures/deeplift_score_comparison.pdf")
    plt.close()

    #df4 = df3.sort_values(by='q-value').drop_duplicates(subset='Target_ID', keep='first')

    sns.set_style(style='white')
    plt.figure(figsize=(11, 6))
    sns.scatterplot(x='logInfl', y='log_qval', hue='Reproducibility', palette='rainbow', s=40, linewidth=0, legend='full', data=df2)  #plt.axvline(x=0.5)
    plt.axhline(y=3.0)
    plt.xlabel("Log of Influence", fontsize=15)
    plt.ylabel("Negative Log q-value of Best Match", fontsize=15)

    #label some points
    texts=[]
    for i in range(df2.shape[0]):
        x = df2.iloc[i]['logInfl']
        y = df2.iloc[i]['log_qval']
        tf_name = df2.iloc[i]['Filter_Name']
        if not pd.isna(tf_name):
            print(tf_name)
            texts.append(plt.text(x, y, str(tf_name), fontsize=7))

    #adjust_text(texts, only_move={'points':'y', 'text':'y'})

    plt.savefig("./figures/basset_score_comparison.pdf")
    plt.close()
