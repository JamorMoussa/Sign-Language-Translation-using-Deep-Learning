import torch, torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from typing import Any 
import os.path as osp
import json
import os

from asl.plot import plot_accuracy, plot_loss

def train_step_gcn_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    loader: DataLoader, 
    device: torch.device,
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
    lr_schedular: optim.lr_scheduler.LRScheduler,
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

        lr_schedular.step()

        lr = opt.param_groups[0]['lr']

        bar.set_description(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc:{train_acc:.3f} | Test Acc: {test_acc:.3f} | lr: {lr:.5f}")

    return results


def generate_report_for_gcn_model(
    root: str,
    model: nn.Module,
    results: dict[str, list],
    configs: dict[str, Any],
):
    
    reports_path = osp.join(root, "reports", "gcn")

    if not osp.exists(reports_path): os.mkdir(reports_path)

    index = len(os.listdir(reports_path))

    train_path = osp.join(reports_path, f"train_{str(index)}")

    if not osp.exists(train_path): os.mkdir(train_path)

    with open(osp.join(train_path, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    torch.save(model.state_dict(), osp.join(train_path, f"GCNModel_train-acc_{results['train_acc'][-1]:.3f}_test-acc_{results['test_acc'][-1]:.3f}.pt"))
    
    with open(osp.join(train_path, f"results_gcn.json"), "w") as f:
        json.dump(results, f)

    with open(osp.join(train_path, f"configs_gcn.json"), "w") as f:
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
    opt_name: str,
    lr_schedular: optim.lr_scheduler.LRScheduler,
    lr: float,
    epochs: int

): 
    configs = {
        "gcn": gcn_configs,
        "loader": {
            "train_batch": batches[0],
            "test_batch": batches[1],
        },
        "opt": {
            "name": opt_name,
            "lr": lr,
            "lr_schedular": {
                "name": lr_schedular.__class__.__name__,
                "gamma": lr_schedular.gamma,
                "step size": lr_schedular.step_size,
            },
        },
        "epochs": epochs
    }

    return configs