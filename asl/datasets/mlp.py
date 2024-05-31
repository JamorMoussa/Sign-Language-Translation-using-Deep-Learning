import os
import os.path as osp
import torch
from torch.utils.data import Dataset
from .gcn import AslGCNDataset

class AslMLPDataset(Dataset):
    
    def __init__(self, root, gcn_path: str = None):
        
        self.root = root 
        
        self.save_path = osp.join(self.root, "data.pt")
        
        if gcn_path != None:
            dataset = AslGCNDataset(root=gcn_path)
            self.data, self.label = self.construct_data(dataset)
            self.save_processed_data(self.data, self.label)
        
        if os.path.exists(self.root):
            self.data, self.label = self.load_processed_data()
        
    def construct_data(self, dataset):
        X = []
        y = []
        for data in dataset:
            X.append(data.x.flatten())
            y.append(data.y)
        
        data = torch.stack(X, dim=0)
        label = torch.stack(y, dim=0)
        return data, label
    
    def save_processed_data(self, data, label):
        torch.save((data, label), self.save_path)
    
    def load_processed_data(self):
        return torch.load(self.save_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

