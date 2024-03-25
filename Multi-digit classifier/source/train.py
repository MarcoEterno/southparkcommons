import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from source.config import get_system_device, checkpoints_path, logs_path, fast_training
from source.image_classifier import ImageClassifier


def load_model(clf: ImageClassifier, start_epoch=0, checkpoints_dir=checkpoints_path):
    """
    Loads the model and optimizer state if resuming training.
    :param clf: the model to load the state into
    :param start_epoch: if set to n, loads the model from checkpoint_{n-1}.pt
    :param checkpoints_dir:
    :return:
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found, creating it, and loading the model randomly initialized")
        os.mkdir(checkpoints_dir)
        return clf, 0
    if os.path.exists(checkpoints_dir):
        # find the last checkpoint that we saved and load it
        for i in range(start_epoch, 0, -1):
            tentative_last_checkpoint_path = os.path.join(checkpoints_dir,
                                                          f"{clf.numbers_to_recognize}_digit{'s' if clf.numbers_to_recognize != 1 else ''}"
                                                          f"_epoch_{i}_{'fast' if fast_training else 'high_perf'}.pt")
            if os.path.exists(tentative_last_checkpoint_path):
                print(f"Loading the model from : checkpoint_{i}.pt. ")
                checkpoint = torch.load(tentative_last_checkpoint_path)
                clf.load_state_dict(checkpoint['model_state_dict'])
                clf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                return clf, start_epoch
        # if no checkpoints are found, load the model randomly initialized
        print(f"No checkpoints found, loading the model randomly initialized")
        return clf, 0


def save_model(clf: ImageClassifier, epoch, checkpoints_dir=checkpoints_path):
    checkpoint_path = os.path.join(checkpoints_dir,
                                   f"{clf.numbers_to_recognize}_digit{'s' if clf.numbers_to_recognize != 1 else ''}"
                                   f"_epoch_{epoch}_{'fast' if fast_training else 'high_perf'}.pt")
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': clf.state_dict(),
            'optimizer_state_dict': clf.optimizer.state_dict(),
            'loss': clf.loss,
        },
        checkpoint_path
    )


def decide_if_model_needs_saving(clf: ImageClassifier, epoch, save_checkpoint_every_n_epochs, save,
                                 checkpoints_dir=checkpoints_path):
    return epoch % save_checkpoint_every_n_epochs


def train_model(clf: ImageClassifier, training_dataloader, validation_dataloader, start_epoch=0, max_total_epochs=10,
                checkpoints_dir=checkpoints_path,
                device=get_system_device(), save_checkpoint_every_n_epochs=5):
    # TODO: parallelize the training loop for cpu only training

    if max_total_epochs == 0:
        print("No epochs to train for. Exiting training stage.")
        return

    # Create a SummaryWriter instance.
    writer = SummaryWriter(log_dir=logs_path)
    # To access TensorBoard, run the following command in terminal:
    # tensorboard --logdir=logs/fit

    # If the device is CPU and there are multiple cores, use DataParallel
    # if a gpu is available, DistributedDataParallel is used instead
    # for now this is on hold since it modifies the model's structure
    # if device == 'cpu' and torch.get_num_threads() > 1:
    # clf = torch.nn.DataParallel(clf)

    print(f"Training the model for {max_total_epochs} epochs,"
          f" saving checkpoints every {save_checkpoint_every_n_epochs if save_checkpoint_every_n_epochs > 1 else ''} epoch{'s.' if save_checkpoint_every_n_epochs > 1 else '.'}")
    for epoch in range(start_epoch, start_epoch + max_total_epochs):
        # Used to calculate the accuracy
        total_predictions = 0
        correct_predictions = 0

        # Time the training loop
        start_time = time.perf_counter_ns()

        for batch in training_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = clf(x)
            loss = clf.loss(output, y)

            # Backpropagation
            clf.optimizer.zero_grad()
            loss.backward()
            clf.optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_predictions += y.size(0)
            correct_predictions += (predicted == y).sum().item()

        train_accuracy = correct_predictions / total_predictions

        # evaluate model on validation dataset
        clf.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_val_predictions = 0
            correct_val_predictions = 0
            for val_batch in validation_dataloader:
                x_val, y_val = val_batch
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                output_val = clf(x_val)
                _, predicted_val = torch.max(output_val.data, 1)
                total_val_predictions += y_val.size(0)
                correct_val_predictions += (predicted_val == y_val).sum().item()

            val_accuracy = correct_val_predictions / total_val_predictions

            # Log accuracy and other metrics to tensorboard
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Time", round((time.perf_counter_ns() - start_time) / 1e9, 3), epoch)
            writer.add_scalar("Learning rate", clf.optimizer.param_groups[0]['lr'], epoch)
        print(
            f"Epoch: {epoch} - Train Accuracy: {round(train_accuracy, 4)} - Validation Accuracy: {round(val_accuracy, 4)}"
            f" - Time: {round((time.perf_counter_ns() - start_time) / 1e9, 3)}s")

        clf.train()  # Set model back to training mode

        # Save model and optimizer state after save_checkpoint_every_n_epochs epochs
        if epoch % save_checkpoint_every_n_epochs == 0:
            save_model(clf, epoch, checkpoints_dir)

    # Close the writer instance
    writer.flush()
    writer.close()
