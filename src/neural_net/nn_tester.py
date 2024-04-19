# Heather Fryling
# Northeastern University

import torch
import os

class NeuralNetTester:
    '''
    Class to test a Pytorch neural network. If a save path is given, it will automatically
    save a checkpoint for the highest accuracy epoch.
    '''

    def __init__(self, save_path=None):
        self.best_accuracy = 0.0
        self.lowest_loss = float('inf')
        self.save_path = save_path

    # Function to test the network.
    def test(self, epoch, model, optimizer, test_loader,  loss_fn, test_losses, test_accuracy, filter=False, validation=False):
        label = 'val' if validation else 'test'
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        current_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizier_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'accuracy': correct
        }
        if self.save_path:
            if correct > self.best_accuracy:
                self.best_accuracy = correct
                torch.save(current_state, os.path.join(self.save_path, f'best_{label}_acc_model.pt'))
            if test_loss < self.lowest_loss:
                self.lowest_loss = test_loss
                torch.save(current_state, os.path.join(self.save_path, f'best_{label}_loss_model.pt'))
        test_accuracy.append(correct)
        test_losses.append(test_loss)
        if filter:
            heuristic_epoch = 49
            threshold = .6
            if epoch == heuristic_epoch and correct < threshold:
                raise Exception(f'Failure to train. Accuracy reached only {(100*correct):>0.1f} after {heuristic_epoch + 1} epochs.')
        print(f"{label} error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
