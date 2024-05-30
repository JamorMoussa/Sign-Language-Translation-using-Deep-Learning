import matplotlib.pyplot as plt
import numpy as np

import os.path as osp


def plot_base(
    root: str,
    arr1: list,
    arr2: list,
    fig_name: str
):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    plt.plot(arr1)
    plt.plot(arr2)

    plt.savefig(osp.join(root, f"{fig_name}.png"))
    plt.clf() 


def plot_loss(
    root: str,
    train_loss: list,
    test_loss: list,
):
    
    plot_base(
        root= root,
        arr1= train_loss,
        arr2= test_loss,
        fig_name= "loss_plot"
    )

def plot_accuracy(
    root: str,
    train_acc: list,
    test_acc: list,
):
    
    plot_base(
        root= root,
        arr1= train_acc,
        arr2= test_acc,
        fig_name= "accuracy_plt"
    )
