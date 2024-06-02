import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import  transforms
from torch_geometric.loader import DataLoader

from asl import models, datasets
from asl.train.cnn import train_cnn_model, generate_report_for_cnn_model, get_general_configs

import os.path as osp
import argparse
import json
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Train cnn model for ASL.")

    parser.add_argument('--root', type= str, default="./",
                        help="")
    
    parser.add_argument('--dataset', type= str, default="./asl_data/asl_dataset",
                        help="")
    
    parser.add_argument('--train_batch', type= int, default= 100,
                        help="")

    parser.add_argument('--test_batch', type= int, default= 1,
                        help="")
    
    parser.add_argument('--lr', type= float, default= 0.01,
                        help="")

    parser.add_argument('--epochs', type= int, default= 15,
                        help="")
    
    parser.add_argument('--C', type= bool, default= True ,
                        help="")

    parser.add_argument('--stepsize', type= int, default= 1,
                        help="")
    

    parser.add_argument('--gamma', type= float, default= 0.1,
                        help="")

    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ASLCNNDataset(
        args.dataset  ,transform= transform
    )

    trainset, testset = random_split(dataset,  lengths=[0.87, 0.13])

    train_loader = DataLoader(trainset, batch_size= args.train_batch, shuffle=True)
    test_loader = DataLoader(testset, batch_size= args.test_batch, shuffle=True)

    cnn_configs = models.CNNTinyVGGModelConfigs.get_defaults()

    cnn_configs.conv_layers =[
        (3, 10, 3, 1), 
        (10, 10, 3, 1),
        (10, 10, 3, 1),
        (10, 10, 3, 1)
    ]
    cnn_configs.fc_layers = [128, 64, 32]

    model = models.CNNTinyVGG(configs= cnn_configs).to(cnn_configs.device)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(), lr= args.lr)

    lr_schedular = optim.lr_scheduler.StepLR(optimizer= opt,
                                             step_size= args.stepsize, 
                                             gamma= args.gamma)

    results = train_cnn_model(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        lr_schedular= lr_schedular,
        train_loader= train_loader,
        test_loader= test_loader,
        device= cnn_configs.device,
        epochs= args.epochs
    )

    configs = get_general_configs(
        cnn_configs= cnn_configs.to_dict(),
        batches = (args.train_batch,args.test_batch),
        opt_name= opt.__class__.__name__,
        lr_schedular = lr_schedular,
        lr= args.lr,
        epochs= args.epochs
    )

    generate_report_for_cnn_model(
        root= args.root,
        model= model, 
        results= results,
        configs= configs,
    )


