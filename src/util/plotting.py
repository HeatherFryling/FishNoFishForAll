# Heather Fryling
# Functions to plot neural net results.

import matplotlib.pyplot as plt

def plot_loss(val_loss, training_loss, train_x, epoch_size, fpath=None):
    '''
    Plots training loss.
    val_loss: list of validation losses
    training_loss: list of training losses
    train_x: list of training epochs
    epoch_size: number of examples per epoch
    fpath (optional): file path for saving the plot
    '''
    fig = plt.figure()
    val_x = [epoch_size * i for i in range(len(val_loss))]
    plt.plot(train_x, training_loss, color='blue', label='training loss')
    plt.scatter(val_x, val_loss, s=10, c=['red'], label='validation loss')
    plt.legend()
    plt.xlabel('number of training examples seen')
    plt.ylabel('loss')
    if fpath:
      plt.savefig(fpath)
    plt.show()


def plot_accuracy(val_accuracy, training_accuracy, train_x, epoch_size, fpath=None):
    '''
    Plots training accuracy.
    val_loss: list of validation accuracies
    training_loss: list of training accuracies
    train_x: list of training epochs
    epoch_size: number of examples per epoch
    fpath (optional): file path for saving the plot
    '''
    fig = plt.figure()
    val_x = [epoch_size * i for i in range(len(val_accuracy))]
    plt.plot(train_x, training_accuracy, color='blue', label='training accuracy')
    plt.scatter(val_x, val_accuracy, s=10, c=['red'], label='validation accuracy')
    plt.legend()
    plt.xlabel('number of training examples seen')
    plt.ylabel('accuracy')
    if fpath:
      plt.savefig(fpath)
    plt.show()