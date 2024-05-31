import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from asl import models, datasets
from asl.train.mlp import train_mlp_model, generate_report_for_mlp_model, get_general_configs

import os.path as osp
import argparse
import json
import os



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Train MLP model for ASL.")

    parser.add_argument('--root', type= str, default="./",
                        help="")
    
    parser.add_argument('--train_batch', type= int, default= 42,
                        help="")

    parser.add_argument('--test_batch', type= int, default= 42,
                        help="")
    
    parser.add_argument('--lr', type= float, default= 0.01,
                        help="")

    parser.add_argument('--epochs', type= int, default= 30,
                        help="")
    
    parser.add_argument('--dataset', type= str, default= "data/mlp/dataset",
                        help="")

    args = parser.parse_args()

    dataset = datasets.AslMLPDataset(
        root= osp.join(args.root, args.dataset)
    )

    trainset, testset = random_split(dataset,  lengths=[0.87, 0.13])

    train_loader = DataLoader(trainset, batch_size= args.train_batch, shuffle=True)
    test_loader = DataLoader(testset, batch_size= args.test_batch, shuffle=True)

    mlp_configs = models.MLPConfigs.get_defaults()

    mlp_configs.mlp_layers = [(42, 128), (128, 28)]
    # gcn_configs.fc_layers = [(64, 32), (32, 32)]

    model = models.MLPModel(configs= mlp_configs).to(mlp_configs.device)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr= args.lr)

    results = train_mlp_model(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        train_loader= train_loader,
        test_loader= test_loader,
        device= mlp_configs.device,
        epochs= args.epochs
    )

    configs = get_general_configs(
        gcn_configs= mlp_configs.to_dict(),
        batches = (args.train_batch,args.test_batch),
        lr= args.lr,
        epochs= args.epochs
    )

    generate_report_for_mlp_model(
        root= args.root,
        model= model, 
        results= results,
        configs= configs,
    )


