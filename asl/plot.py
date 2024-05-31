import matplotlib.pyplot as plt
import numpy as np

import os.path as osp


def plot_base(
    root: str,
    arr1: list,
    arr2: list,
    fig_name: str,
    title: str,
    x_label: str, 
    y_label: str,
    plot_label1: str,
    plot_label2: str,
):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    plt.plot(arr1, label= plot_label1)
    plt.plot(arr2, label= plot_label2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.legend(loc="upper left")
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
        fig_name= "loss_plot",
        title= "Training and Testing Loss",
        x_label= "epochs",
        y_label= "loss",
        plot_label1= "Training Loss",
        plot_label2= "Testing Loss",
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
        fig_name= "accuracy_plt",
        title= "Training and Testing Accuracy",
        x_label= "epochs",
        y_label= "Accuracy",
        plot_label1= "Training Accuracy",
        plot_label2= "Testing Accuracy",
    )
