import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from mof2zeo import __root_dir__
from sklearn.preprocessing import StandardScaler

class CSVDataset(Dataset):
    def __init__(self, direc, scaled = True, scaler = None, feature_name_dir =f'{__root_dir__}/data/feature_name.txt'):
        super().__init__()

        # import CSV file
        self.direc = Path(direc).resolve()
        if not self.direc.exists():
            raise ValueError(f'{direc} does not exists.')
        self.data = pd.read_csv(direc)

        with open(feature_name_dir, 'r') as g:
            self.feature_names = [line.strip() for line in g.readlines()]
            
        # topology, node, edge class dictionary
        self.topo_dict = TOPO_DICT
        self.node_dict = NODE_DICT
        self.edge_dict = EDGE_DICT

        self.x = self.data[self.feature_names].to_numpy()  #(self.data.iloc[:, :-1].to_numpy())
        self.y = self.make_target_data() 

        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
            

        if scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        # dimension: (D) -> (D,1)
        self.x = self.x.unsqueeze(-1) 

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return [x, y]
    
    def __len__(self):
        return len(self.x)

    def make_target_data(self):
        target_list = []
        split_list = [fname.split('.')[0].split('+') for fname in self.data['filename']]

        for splits in split_list:
            # 무조건 topo + N + E 만 허용
            if len(splits) != 3:
                continue  # 또는 raise ValueError

            topo, n, e = splits
            if (not n.startswith('N')) or (not e.startswith('E')):
                continue

            target_num = [self.topo_dict[topo], self.node_dict[n], self.edge_dict[e]]
            target_list.append(target_num)

        return torch.tensor(target_list)


class MOFGenDataset(Dataset):
    def __init__(self, direc, scaled = True, scaler = None, feature_name_dir =f'{__root_dir__}//data/feature_name.txt' ):
        super().__init__()

        # import CSV file
        self.direc = Path(direc).resolve()
        if not self.direc.exists():
            raise ValueError(f'{direc} does not exists.')
        self.data = pd.read_csv(direc).dropna()

        with open(feature_name_dir, 'r') as g:
            self.feature_names = [line.strip() for line in g.readlines()]
            
        # topology, node, edge class dictionary
        self.topo_dict = TOPO_DICT
        self.node_dict = NODE_DICT
        self.edge_dict = EDGE_DICT

        self.y = self.make_target_data() 
 

    def __getitem__(self, idx):
        y = self.y[idx]

        return y
    
    def __len__(self):
        return len(self.y)

    def make_target_data(self):
        target_list = []
        split_list = [fname.split('.')[0].split('+') for fname in self.data['filename']]

        for splits in split_list:
            # 무조건 topo + N + E 만 허용
            if len(splits) != 3:
                continue

            topo, n, e = splits
            if (not n.startswith('N')) or (not e.startswith('E')):
                continue

            target_num = [self.topo_dict[topo], self.node_dict[n], self.edge_dict[e]]
            target_list.append(target_num)

        return torch.tensor(target_list)



        

def make_class(file_name):
    with open(file_name, ) as f:
        lines = f.readlines()
    class_dict = {name.strip(): i for i, name in enumerate(lines)}
    return class_dict




class Scaler(object):
    def __init__(self, mean, std, target_mean, target_std, eps: float = 1e-6):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps

    def to(self, dtype=None, device=None):
        self.mean = self.mean.to(dtype=dtype, device=device)
        self.std = self.std.to(dtype=dtype, device=device)

    def encode(self, batch):
        if isinstance(batch, torch.Tensor):
            return (batch - self.mean) / (self.std + self.eps)
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            return (batch - mean) / (std + self.eps)
    
    def encode_target(self, target):
        return (target - self.target_mean) / (self.target_std + self.eps)

    def decode(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch * self.std + self.mean
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            return batch * std + mean
    
    def decode_target(self, target):
        
        return target * self.target_std + self.target_mean


TOPO_DICT = make_class(f'{__root_dir__}/data/topology.txt')
NODE_DICT = make_class(f'{__root_dir__}/data/node.txt')
EDGE_DICT = make_class(f'{__root_dir__}/data/edge.txt')
