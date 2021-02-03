import torch
from torch.utils.data import DataLoader
from pyfaidx import Fasta
import pandas as pd

from RegNet import utils


class SeqData(torch.utils.data.IterableDataset):
    """Dataloader for sequence data. Compatible with torch DataLoader"""

    def __init__(self, genome: str, data_path: str, window: int = 1000):
        super().__init__()
        self.genome = Fasta(genome, as_raw=True)
        self.matrix = pd.read_table(data_path, sep="\t")
        self.window = window
    
    def __iter__(self):
        for row in self.matrix.iterrows():
            coords = row[0].split("-")
            startpos = int(coords[1])
            endpos = int(coords[2])
            width = endpos - startpos
            midpoint = startpos + int(width / 2)
            region_start = midpoint - int(self.window / 2)
            region_end = midpoint + int(self.window / 2)
            fasta = self.genome[coords[0]][region_start:region_end]
            hot = utils.one_hot(fasta)
            yield(hot, torch.tensor(row[1]))


class eQTLData(torch.utils.data.IterableDataset):
    """Dataloader for sequence data. Used for checking eQTL peaks, does not load the labels"""
    
    def __init__(self, genome: str, data_path: str, window: int = 1000):
        super().__init__()
        self.genome = Fasta(genome, as_raw=True)
        self.matrix = pd.read_table(data_path, sep="\t")
        self.window = window
    
    def __iter__(self):
        for row in self.matrix.iterrows():
            coords = row[0].split("-")
            startpos = int(coords[1])
            endpos = int(coords[2])
            width = endpos - startpos
            midpoint = startpos + int(width / 2)
            region_start = midpoint - int(self.window / 2)
            region_end = midpoint + int(self.window / 2)
            fasta = self.genome[coords[0]][region_start:region_end]
            hot = utils.one_hot(fasta)
            positions = [int(x) for x in row[1][0].split(',')[:-1]] # extract vector of eqtl positions from string
            yield(hot, positions)
    
    

