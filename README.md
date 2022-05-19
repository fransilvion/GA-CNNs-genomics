# GA Generated Sequence Interpretation

The goal of this repo is to store techniques for interpreting and extracting motifs from GA-generated DNA sequences.

This is the joint project with Etienne Meunier (https://github.com/Etienne-Meunier)

The project was presented at MLCB NeurIPS 2019 (poster):

Etienne Meunier, German Novakovsky and Sara Mostafavi. **Interpreting deep learning models in genomics using genetic algorithms**

(https://sites.google.com/cs.washington.edu/mlcb2019/presentations/posters)

## How to use ? 

- Store an umcompressed folder containing a subfolder population itself containing .fa generated file ( one file by class ) in the base of the project 

- ``./run_motif_scan name_of_folder``  will launch the jaspar run 

- ``cd python3 Get_TFs/get_TFs.py name_of_folder`` extract TF from jaspar run

- ``python3 Count_Sashas_Motif.py name_of_folder`` extract a count of the filter in data/filter_motifs_pwm.meme'
