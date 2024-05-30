import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from asl import models, datasets
from asl.train.gcn import train_gcn_model

import os.path as osp
import argparse
import json
import os



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description= "Train GCN model for ASL.")

    parser.add_argument('--root', type= str, default="./",
                        help="")
    
    parser.add_argument('--train_batch', type= int, default= 500,
                        help="")

    parser.add_argument('--test_batch', type= int, default= 100,
                        help="")
    
    parser.add_argument('--lr', type= float, default= 0.01,
                        help="")

    parser.add_argument('--epochs', type= int, default= 1,
                        help="")

    args = parser.parse_args()

    dataset = datasets.AslGCNDataset(
        root= osp.join(args.root, "data/gcn/dataset")
    )

    trainset, testset = random_split(dataset,  lengths=[0.87, 0.13])

    train_loader = DataLoader(trainset, batch_size=500, shuffle=True)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)

    gcn_configs = models.GCNModelConfigs.get_defaults()

    model = models.GCNModel(configs= gcn_configs).to(gcn_configs.device)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr= args.lr)

    results = train_gcn_model(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        train_loader= train_loader,
        test_loader= test_loader,
        device= gcn_configs.device,
        epochs= args.epochs
    )

    length = len(os.listdir(osp.join(args.root, "reports")))

    with open(osp.join(args.root, f"reports/train_gcn{length}.json"), "w") as f:
        json.dump(results, f)
