import torch, torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from typing import Any 
import os.path as osp
import json
import os

from asl.plot import plot_accuracy, plot_loss

def train_step_mlp_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    loader: DataLoader, 
    device: torch.device
) -> None:
    
    for  inputs, labels in loader:
        
        labels = labels.squeeze()
        
        opt.zero_grad()

        inputs = inputs.to(device)
        
        outputs = model(inputs)
        
        labels = labels.to(device)
        
        loss = loss_fn( outputs, labels)

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
    
    for inputs , labels in loader:
        
        labels = labels.to(device)

        labels = labels.squeeze()
                
        inputs= inputs.to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        pred = outputs.argmax(dim=1)
        
        accuracy += int((pred == labels).sum())

        total_loss += loss.item()

    total_loss /= len(loader.dataset)
    accuracy /= len(loader.dataset)

    return total_loss, accuracy


def train_mlp_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device,
    epochs: int = 200
) -> None:

    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }
    for epoch in (bar := tqdm(range(epochs))):

        train_step_mlp_model(
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


def generate_report_for_mlp_model(
    root: str,
    model: nn.Module,
    results: dict[str, list],
    configs: dict[str, Any] 
):
    
    reports_path = osp.join(root, "reports", "mlp")

    if not osp.exists(reports_path): os.mkdir(reports_path)

    index = len(os.listdir(reports_path))

    train_path = osp.join(reports_path, f"train_{str(index)}")

    if not osp.exists(train_path): os.mkdir(train_path)

    with open(osp.join(train_path, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    torch.save(model.state_dict(), osp.join(train_path, f"MLPModel_train-acc_{results['train_acc'][-1]:.3f}_test-acc_{results['test_acc'][-1]:.3f}.pt"))
    
    with open(osp.join(train_path, f"results_mlp.json"), "w") as f:
        json.dump(results, f)

    with open(osp.join(train_path, f"configs_mlp.json"), "w") as f:
        json.dump(configs, f)

    plot_loss(
        root= train_path,
        train_loss= results["train_loss"],
        test_loss= results["test_loss"],
    )

    plot_accuracy(
        root= train_path,
        train_acc= results["train_acc"],
        test_acc= results["test_acc"],
    )


def get_general_configs(
    gcn_configs: dict,
    batches: tuple[int, int],
    lr: float,
    epochs: int

): 
    configs = {
        "mlp": gcn_configs,
        "loader": {
            "train_batch": batches[0],
            "test_batch": batches[1],
        },
        "opt": {
            "lr": lr
        },
        "epochs": epochs
    }

    return configs