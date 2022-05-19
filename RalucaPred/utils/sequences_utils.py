import torch
import numpy as np


def seq_letters_num(sequence, mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}):
    """
    Given a sequence encoded with Nucleotide letters and return it with numbers
    Mapping : A -> 0 , C -> 1, G -> 2 , T -> 3 by default but can be replaced
    Param : sequence : sequence with numbers (list, np.array, torch.tensor)
    return : sequence with letters (string)
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()
    if isinstance(sequence, np.ndarray):
        sequence = sequence.copy().astype('object')
    num_letters = mapping
    for k in num_letters.keys():
        sequence[sequence == k] = num_letters[k]
    return sequence


def seq_one_hot(sequence):
    """
    Take a DNA sequence with numbers and turn it to one hot sequence
    0 -> 1000 , 1-> 0100, 2-> 0010, 3-> 0001
    param sequence : sequence (1,N)
    return one_hot sequence (N,4)
    """
    y = torch.eye(4)
    return(y[sequence])

