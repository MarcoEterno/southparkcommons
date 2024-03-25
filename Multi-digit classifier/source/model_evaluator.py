"""
The module serves the purpose of evaluating multiple model checkpoints, in order to find the best one.
"""

import os
from collections import defaultdict

import torch
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt

from config import get_system_device, checkpoints_path, custom_data_path, fast_training
from data_loader import get_n_digits_train_validation_test_dataset
from plot_data import plot_model_inference
from train import train_model, load_model
from image_classifier import ImageClassifier
from inference import test_model_performance
from plot_data import plot_dataset

max_number_of_epochs = 300
device = get_system_device(print_info=True)

def load_predifined_checkpoint(clf: ImageClassifier, predefined_checkpoints_path, epoch):
    """
    Loads a predefined checkpoint on the input model
    :param clf: the model to load the checkpoint into
    :param predefined_checkpoints_path: the path to the checkpoint
    :param epoch: the epoch of the checkpoint
    :return: the model with the checkpoint loaded
    """
    checkpoint = torch.load(predefined_checkpoints_path)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return clf

def plot_accuracy(accuracy, n_digits_in_number_to_classify):
    """
    Plots the accuracy for each epoch
    :param accuracy: a dictionary containing as key the epoch and as value the accuracy
    :param n_digits_in_number_to_classify: the number of digits to classify

    """
    plt.plot(accuracy.keys(), accuracy.values())
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy for {n_digits_in_number_to_classify} digits")
    plt.savefig(os.path.join(custom_data_path, f"accuracy_{n_digits_in_number_to_classify}_digit{'s'if n_digits_in_number_to_classify!=1 else ''}.png"))
    plt.show()

def explore_network_weights(clf: ImageClassifier):
    """
    Prints an overview of the network's weights
    :param clf: the model to print the weights of
    """
    total_params = 0
    for layer in clf.model:
        print(layer)
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            print(layer.weight.shape)
            print(layer.bias.shape)
            print(layer.weight.data.shape)
            print(layer.bias.data.shape)

            #calculating the totl number of parameters
            n_params = layer.weight.numel() + layer.bias.numel()
            print(f"Number of parameters in layer {layer}: {n_params}")
            total_params += n_params
    print(f"Total number of parameters:  {total_params}")
def evaluate_all_chepoints(n_digits_of_interest = range(1, 5)):
    """
    This function evaluates all checkpoints for each number of digits to classify.
    It also plots the accuracy for each number of digits to classify.
    :param range: the range of digits to classify we are interested with.
    The range is inclusive on the left and exclusive on the right

    """
    for n_digits_in_number_to_classify in n_digits_of_interest:
        accuracy = defaultdict(list)
        best_model_epoch = defaultdict(int)
        print(f"Evaluating model for {n_digits_in_number_to_classify} digits")

        # Load datasets
        _, __, test_datasets = get_n_digits_train_validation_test_dataset(n=n_digits_in_number_to_classify, augment_data=True, scale_data_linearly=True)

        # Create model
        clf = ImageClassifier(n_digits_to_recognize=n_digits_in_number_to_classify, optimizer=Adam,
                              loss_fn=nn.CrossEntropyLoss, lr=1e-3, print_info=False).to(device)

        for epoch in range(0, max_number_of_epochs):
            # Load predefined checkpoint if available
            if os.path.exists(os.path.join(checkpoints_path,
                                           f"{clf.numbers_to_recognize}_digit{'s' if clf.numbers_to_recognize != 1 else ''}"
                                           f"_epoch_{epoch}_{'fast' if fast_training else 'high_perf'}.pt")):
                clf, epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=epoch)

                # Test model
                accuracy[epoch] = test_model_performance(clf, test_datasets)

        # Plot accuracy
        plot_accuracy(accuracy, n_digits_in_number_to_classify)
        # Find best model
        best_model_epoch[n_digits_in_number_to_classify] = max(accuracy, key=accuracy.get)
        print(f"Best accuracy for {n_digits_in_number_to_classify} digits: {max(accuracy.values())}, achieved at epoch {best_model_epoch[n_digits_in_number_to_classify]-1}")

        # Plotting inference from the best model
        clf, epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=best_model_epoch[n_digits_in_number_to_classify]-1)
        plot_model_inference(clf, test_datasets)

if __name__ == '__main__':

    clf = ImageClassifier(n_digits_to_recognize=4)
    #explore_network_weights(clf)
    evaluate_all_chepoints(range(1, 5))

"""
Keep track of best models:
- 1 digit: 99.6% accuracy, achieved at epoch 7
- 2 digits: 98.8% accuracy, achieved at epoch 9
- 3 digits: 93.7% accuracy, achieved at epoch 18
- 4 digits: 36.7% accuracy, achieved at epoch 20
in 3 and four digits the accuracy is low due to the data generation process. 
In fact, number should be generated:
1. by shuffling the digits and not concatenating them
2. by increasing the numbers of samples with increasing digits to classify.
"""

