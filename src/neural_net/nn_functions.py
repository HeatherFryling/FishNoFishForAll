# Heather Fryling
# Northeastern University

import torch


# function to train the network
def train(epoch, epoch_size, network, train_loader, optimizer, loss_fn, log_interval, train_losses, train_counter, train_accuracy):

    # make sure the network is in training mode so the dropout layers work correctly
    network.train()  # opposite is network.eval()

    examples_seen = epoch * epoch_size
    correct = 0
    loss = None
    for batch_idx, (data, target) in enumerate( train_loader ):
        optimizer.zero_grad() # zero the gradients for each batch
        output = network( data ) # forward pass through the network
        loss = loss_fn( output, target ) # compute the loss
        loss.backward() # backward pass to calculate the gradients
        optimizer.step() # modify the weights
        correct += (output.argmax(1) == target).type(torch.float).sum().item()

        if batch_idx % log_interval == 0:
            print( "Training epoch: %d  batch %d %4d/%4d  loss %.5f" % ( epoch,
                                                                      batch_idx,
                                                                      batch_idx * len(data),
                                                                      len(train_loader.dataset),
                                                                      loss.item() ) )
            train_losses.append( loss.item() )
            train_counter.append( examples_seen )
            if batch_idx != 0:
                train_accuracy.append(correct / (len(data) * log_interval))
            else:
                train_accuracy.append(correct / len(data))
            correct = 0
        examples_seen += len(data)

    return
