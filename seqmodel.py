import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pyfaidx import Fasta
import pandas as pd
import numpy as np
from argparse import ArgumentParser


# TODO: should pre-compute 1hot encoding for whole genome, then store in compressed indexed format and read in as needed. Or store whole thing in memory.
def one_hot(string, bases = {'A':0, 'C':1, 'G':2, 'T':3}):
    """Convert DNA sequence to 1-hot array"""
    res = np.zeros( (4,len(string)), dtype=np.float32 )
    for j in range(len(string)):
        if string[j] in bases:
            res[ bases[ string[j] ], j ]=float(1.0)
    return(res)


def get_final_layer_input_size(in_width, pool_sizes, n_kernels):
    """Return size of final layer after series of same convolutions with max pooling between"""
    out_size = in_width
    for i in range(len(pool_sizes)):
        out_size = int(out_size / pool_sizes[i])
    out_size = out_size * n_kernels[i]
    return out_size


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
            hot = one_hot(fasta)
            hot = torch.tensor(hot)
            yield(hot, torch.tensor(row[1]))


class Exponential(nn.Module):
    """Exponential activation function for use with nn.Sequential"""
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return torch.exp(input)


class SeqModel(pl.LightningModule):
    """Convolutional neural network to predict normalized accessibility from sequence, for different cell types"""

    def __init__(
        self,
        objective = F.mse_loss,
        in_width: int = 500,
        n_kernels: list = [300, 128, 128],
        pool_sizes: list = [1, 4, 4],
        kernel_size: int = [11, 7, 5],
        dropout_conv: float = 0.1,
        dropout_linear: float = 0.2,
        n_hidden: int = 1000,
        n_class: int = 15,
        learning_rate: float = 2e-03,
        l2: float = 1e-06
    ):
        super().__init__()
        
        out_size = get_final_layer_input_size(
            in_width=in_width,
            pool_sizes=pool_sizes,
            n_kernels=n_kernels
        )
        # TODO: expand conv layers as list and construct network from list of layers, avoids duplication here
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=n_kernels[0],
                kernel_size=kernel_size[0],
                stride=1,
                padding=int((kernel_size[0]-1)/2)
            ),
            nn.BatchNorm1d(n_kernels[0]),
            Exponential(),
            nn.MaxPool1d(kernel_size=pool_sizes[0], stride=pool_sizes[0]),
            nn.Dropout2d(dropout_conv),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_kernels[0],
                out_channels=n_kernels[1],
                kernel_size=kernel_size[1],
                stride=1,
                padding=int((kernel_size[1]-1)/2)
            ),
            nn.BatchNorm1d(n_kernels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_sizes[1], stride=pool_sizes[1]),
            nn.Dropout2d(dropout_conv),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_kernels[1],
                out_channels=n_kernels[2],
                kernel_size=kernel_size[2],
                stride=1,
                padding=int((kernel_size[2]-1)/2)
            ),
            nn.BatchNorm1d(n_kernels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_sizes[2], stride=pool_sizes[2]),
            nn.Dropout2d(dropout_conv)
        )
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=out_size, out_features=n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_linear)
        )
        self.linear2 = nn.Linear(in_features=n_hidden, out_features=n_class)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.objective = objective
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.FloatTensor)
        y_hat = self(x).type(torch.FloatTensor)
        loss = self.objective(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.FloatTensor)
        y_hat = self(x).type(torch.FloatTensor)
        loss = self.objective(y_hat, y)
        self.log('validation_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.l2)


def main(args):
    # logger
    logger = pl_loggers.CometLogger(
        api_key="CqZ4OZlEj1rN92susoom7kSjG",
        project_name="hparam_tests",
        experiment_name=args.exp_name
    )
    
    # model and trainer    
    model = SeqModel(
        objective=F.mse_loss,
        in_width=1000,
        dropout_conv=0.1,
        dropout_linear=0.2,
        n_hidden=args.n_hidden,
        n_kernels=[300, 200, 200],
        kernel_size=[11, 5, 5],
        pool_sizes=[int(x) for x in args.pool_sizes],
        n_class=15
    )
    trainer = Trainer(gpus=1, logger=logger, max_epochs=10)
    
    # data
    train_data = SeqData(genome="/home/stuartt/atac-cnn/data/hg38.fa", data_path=args.train_data, window=1000)
    val_data = SeqData(genome="/home/stuartt/atac-cnn/data/hg38.fa", data_path=args.val_data, window=1000)
    train_loader = DataLoader(train_data, batch_size=64, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=1)
    
    # run model (saved automatically)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser(description="Convolutional neural network: predict accessibility from sequence")
    parser.add_argument("--n_hidden", help="Number of neurons in final dense layer", type=int, required=True)
    parser.add_argument("--pool_sizes", help="Max pooling size for each conv layer (list)", nargs="+", required=True)
    parser.add_argument("--exp_name", help="Name for the experiment (shown in logger)", type=str, required=True)
    parser.add_argument("--train_data", help="Path to training data", type=str, required=True)
    parser.add_argument("--val_data", help="Path to validation data", type=str, required=True)
    args = parser.parse_args()

    main(args)