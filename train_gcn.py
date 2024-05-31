import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from asl import models, datasets
from asl.train.gcn import train_gcn_model, generate_report_for_gcn_model, get_general_configs

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
    
    parser.add_argument('--dataset', type= str, default= "data/gcn/dataset",
                        help="")

    args = parser.parse_args()

    dataset = datasets.AslGCNDataset(
        root= osp.join(args.root, args.dataset)
    )

    trainset, testset = random_split(dataset,  lengths=[0.87, 0.13])

    train_loader = DataLoader(trainset, batch_size= args.train_batch, shuffle=True)
    test_loader = DataLoader(testset, batch_size= args.test_batch, shuffle=True)

    gcn_configs = models.GCNModelConfigs.get_defaults()

    gcn_configs.gcn_layers = [(2, 64), (64, 64)]
    gcn_configs.fc_layers = [(64, 32), (32, 32)]

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

    configs = get_general_configs(
        gcn_configs= gcn_configs.to_dict(),
        batches = (args.train_batch,args.test_batch),
        lr= args.lr,
        epochs= args.epochs
    )

    generate_report_for_gcn_model(
        root= args.root,
        model= model, 
        results= results,
        configs= configs,
    )


