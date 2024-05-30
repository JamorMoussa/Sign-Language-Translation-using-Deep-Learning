import torch, torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from typing import Any 
import os.path as osp

def train_step_gcn_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    loader: DataLoader, 
    device: torch.device
) -> None: 

    for data in loader:

        opt.zero_grad()

        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)

        loss.backward()

        opt.step()



def compute_loss_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss._Loss,
    device: torch.device
) -> float:
    
    model.eval()

    total_loss = 0
    accuracy = 0
    
    for data in loader:

        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)

        loss = loss_fn(out, data.y)

        pred = out.argmax(dim=1)
        accuracy += int((pred == data.y).sum())

        total_loss += loss.item()

    # total_loss /= len(loader.dataset)
    accuracy /= len(loader.dataset)

    return total_loss, accuracy


def train_gcn_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device,
    epochs: int = 10
) -> None:

    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    for epoch in (bar := tqdm(range(epochs))):

        train_step_gcn_model(
            model= model,
            loss_fn= loss_fn,
            opt= opt,
            loader= train_loader,
            device= device
        )

        train_loss, train_acc = compute_loss_accuracy(
            model= model,
            loader= train_loader,
            loss_fn= loss_fn,
            device= device
        )

        test_loss, test_acc = compute_loss_accuracy(
            model= model,
            loader= test_loader,
            loss_fn= loss_fn,
            device= device
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def generate_report(
    root: str,
    model: nn.Module,
    results: dict[str, list],
    configs: dict[str, Any] 
):
    
    # reports_path = osp.join(r"r")
    pass 
