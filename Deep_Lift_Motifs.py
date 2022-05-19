import pandas as pd
import os
import plotly.graph_objects as go
import seaborn as sns
PDBX = os.path.expanduser('~/Dropbox/UBC/')
#%%
cell_types = {
              'B':['task0','task1','task2','task3','task4','task5','task6','task7','task8','task9','task10','task11','task12','task13','task14','task78','task79','task80'],
              'myeloid':['task15','task16','task17','task18','task19','task20','task27','task28','task29','task30','task31','task32','task33','task34','task37','task38'],
              'innate.lym':['task21','task22','task23','task24','task39','task40','task41','task42','task43','task44'],
              'stem':['task25','task26','task35','task36','task49'],
              'abT':['task45','task46','task47','task48','task50','task51','task52','task53','task54','task55','task56','task57','task58','task59','task60','task61','task62','task63','task64','task72','task73','task74','task75','task76','task77'],
              'gdT':['task65','task66','task67','task68','task69','task70','task71']
              }

inv_cell_types = {}
for k, v in cell_types.items() :
    for e in v :
        inv_cell_types[e] = k

#%%
deep_lift_sasha



deep_lift_sasha = pd.read_csv(PDBX+'data/deeplift_summary_sasha.csv',index_col=0)


deep_lift_sasha['task'] = deep_lift_sasha['Query_ID'].str.split('_').str[0]
deep_lift_sasha['TF_Name'] = deep_lift_sasha['TF_Name'].str.split('/').str[0]
deep_lift_sasha['Group'] = [inv_cell_types[a] for a in deep_lift_sasha['task']]
deep_lift_sasha.groupby(['Group','Target_ID'])['Ave_Score'].mean()
piv = deep_lift_sasha.pivot_table(index='Group',columns='Target_ID',values='Ave_Score',fill_value=0)
deep_lift_sasha[deep_lift_sasha['Target_ID'] == 'filter255']['TF_Name'][0]
px.bar(piv,x='B')
deep_lift_sasha[deep_lift_sasha['Target_ID'] == 'filter10']['TF_Name']
[a for a in piv.columns]
go.Figure(go.Bar(y=piv.loc['B'],x=[deep_lift_sasha[deep_lift_sasha['Target_ID'] == a]['TF_Name'].iloc[0] for a in piv.columns]))
