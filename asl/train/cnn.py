import torch, torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from typing import Any 
import os.path as osp
import json
import os

from asl.plot import plot_accuracy, plot_loss

def train_step_cnn_model(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: optim.Optimizer,
    loader: DataLoader, 
    device: torch.device,
    lr_schedular: optim.lr_scheduler.LRScheduler = None,
) -> None: 

    for i, (img, lbl) in tqdm(enumerate(loader), total=len(loader)):
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred_lbl = model(img)
            loss = loss_fn(pred_lbl, lbl)
            loss.backward()
            opt.step()
            # predicted = pred_lbl.argmax(dim=1)
            # total_train_correct += (predicted == lbl).sum().item()
            # total_train_samples += lbl.size(0)

            if lr_schedular is not None:
              opt.step()
            else:
              lr_schedular.step()



def compute_loss_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss._Loss,
    device: torch.device
) -> float:
    
    model.eval()
    total_loss = 0
    test_correct = 0
    test_samples = 0
    
    for i,(img, lbl) in tqdm(enumerate(loader), total=len(loader)):
        img, lbl = img.to(device), lbl.to(device)

        pred_lbl = model(img)
        loss = loss_fn(pred_lbl, lbl)
    

        total_loss += loss.item()
        test_predicted = pred_lbl.argmax(dim=1)
        test_correct += (test_predicted == lbl).sum().item()
        test_samples += lbl.size(0)
        
    avg_test_loss = loss  / len(loader)
    test_accuracy = test_correct / test_samples * 100

    return avg_test_loss, test_accuracy


def train_cnn_model(
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

        train_step_cnn_model(
            model= model,
            loss_fn= loss_fn,
            opt= opt,
            lr_schedular= lr_schedular,
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

        lr = opt.param_groups[0]['lr']

        bar.set_description(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc:{train_acc:.3f} | Test Acc: {test_acc:.3f} | lr: {lr:.5f}")

    return results


def generate_report_for_cnn_model(
    root: str,
    model: nn.Module,
    results: dict[str, list],
    configs: dict[str, Any],
):
    
    reports_path = osp.join(root)

    if not osp.exists(reports_path): os.mkdir(reports_path)

    index = len(os.listdir(reports_path))

    train_path = osp.join(reports_path, f"train_{str(index)}")

    if not osp.exists(train_path): os.mkdir(train_path)

    with open(osp.join(train_path, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    torch.save(model.state_dict(), osp.join(train_path, f"CNNModel_train-acc_{results['train_acc'][-1]:.3f}_test-acc_{results['test_acc'][-1]:.3f}.pt"))
    
    with open(osp.join(train_path, f"results_ccn.json"), "w") as f:
        json.dump(results, f)

    with open(osp.join(train_path, f"configs_ccn.json"), "w") as f:
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
    cnn_configs: dict,
    batches: tuple[int, int],
    opt_name: str,
    lr_schedular: optim.lr_scheduler.LRScheduler,
    lr: float,
    epochs: int

): 
    configs = {
        "ccn": cnn_configs,
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