import torch
from torch import nn
from torch.optim import Adam

from config import get_system_device, checkpoints_path
from data_loader import get_n_digits_train_validation_test_dataset
from image_classifier import ImageClassifier
from inference import test_model_performance
from plot_data import plot_dataset
from plot_data import plot_model_inference
from train import train_model, load_model

# Hyperparameters
NUM_DIGITS = 2  # number of digits to classify
start_epoch = 0  # if set to n, loads the model from checkpoint_{n}.pt. if checkpoint_{n}.pt does not exist, it will start training from the latest available checkpoint
total_epochs_to_train = 16  # total number of epochs that we want to train for
save_checkpoint_every_n_epochs = 5  # save a checkpoint every n epochs



if __name__ == '__main__':
    device = get_system_device(print_info=True)
    train_dataloader, validation_dataloader, test_dataloader = get_n_digits_train_validation_test_dataset(
        n=NUM_DIGITS,
        augment_data=False,
        scale_data_linearly=True
    )

    # plot dataset
    plot_dataset(train_dataloader)

    # Create model
    clf = ImageClassifier(n_digits_to_recognize=NUM_DIGITS, optimizer=Adam,
                          loss_fn=nn.CrossEntropyLoss, lr=1e-3).to(device)

    # Load model and optimizer state if resuming training
    clf, start_epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=start_epoch)

    # Train
    train_model(
        clf=clf,
        training_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        max_total_epochs=total_epochs_to_train,
        start_epoch=start_epoch,
        device=device,
        save_checkpoint_every_n_epochs=save_checkpoint_every_n_epochs,
        checkpoints_dir=checkpoints_path
    )

    # Test model
    test_model_performance(clf, test_dataloader, device)

    # Plot model inference
    plot_model_inference(clf, test_dataloader, device=device)