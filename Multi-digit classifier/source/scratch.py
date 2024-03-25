import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from config import get_system_device, checkpoints_path, logs_path
from image_classifier import ImageClassifier
from torchvision import transforms

"""
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""


#
#
# TODO: wht is the SVHN dataset?
# TODO: find a strategy to stop training when the model is overfitting, and give the option to the user to choose. may be early stopping
# TODO: add tensorboard to requirements.txt
# TODO: add progress bars to reassure the user while training
# TODO: add in readme how datasets are created and stored
# TODO: GUI?
# TODO: DOCSTRINGS
# TODO: split training functions
# TODO: data augmentation
# TODO: add tests
# TODO: fix the parallelization of the model training
# TODO: bring all params in config sistematically
# TODO: yaml file for config
# TODO: it is possible to separate the training of dense and convolutional layers, and train them separately, but it is not implemented yet
# TODO: increase font boldness in plots


def train_epoch(clf: ImageClassifier, train_dataloader, validation_dataloader, device=get_system_device()):
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    for batch in train_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Find predictions
        outputs = clf(inputs)
        loss = clf.loss(outputs, labels)

        # Backpropagation
        clf.optimizer.zero_grad()
        loss.backward()
        clf.optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    average_training_loss = running_loss / len(train_dataloader)
    accuracy = correct_predictions / total_predictions
    return average_training_loss, accuracy


def validate_epoch(clf: ImageClassifier, dataloader, device=get_system_device()):
    validation_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = clf(inputs)
            loss = clf.loss(outputs, labels)

            validation_loss += loss.item()

    average_validation_loss = validation_loss / len(dataloader)
    return average_validation_loss


def save_checkpoint(model, epoch, checkpoints_dir=checkpoints_path):
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, checkpoint_path)


def load_predifined_checkpoint(clf: ImageClassifier, predefined_checkpoints_path, epoch):
    checkpoint = torch.load(predefined_checkpoints_path)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return clf


def load_latest_available_checkpoint(clf: ImageClassifier, start_epoch=0, checkpoints_dir=checkpoints_path):
    """
    Load the latest available checkpoint from the checkpoints directory
    :param clf: the model to load the checkpoint into
    :param start_epoch: the epoch to load the checkpoint from. if no checkpoint is found for that epoch, the model is
    loaded from the last available checkpoint
    :param checkpoints_dir: the directory where the checkpoints are stored
    :return: clf: the model with the checkpoint loaded into it, start_epoch: the epoch from which the model was loaded
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found, creating it, and loading the model randomly initialized")
        os.mkdir(checkpoints_dir)
        return clf, 0
    if os.path.exists(checkpoints_dir):
        # find the last checkpoint that we saved and load it
        for i in range(start_epoch, 0, -1):
            tentative_last_checkpoint_path = os.path.join(checkpoints_dir,
                                                          f"{clf.numbers_to_recognize}_digit{'s' if clf.numbers_to_recognize != 1 else ''}_epoch_{i}.pt")
            if os.path.exists(tentative_last_checkpoint_path):
                print(f"Loading the model from : checkpoint_{i}.pt. ")
                clf = load_predifined_checkpoint(clf, tentative_last_checkpoint_path, epoch=i)
                return clf, i + 1  # if checkpoint is found, start from the next epoch
        # if no checkpoints are found, load the model randomly initialized
        print(f"No checkpoints found, loading the model randomly initialized")
        return clf, 0


def train_model(model, training_datasets, validation_dataloader, start_epoch=0, max_total_epochs=10,
                checkpoints_dir='checkpoints', save_checkpoint_every_n_epochs=5, device=get_system_device()):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model, start_epoch = load_latest_available_checkpoint(model, start_epoch, checkpoints_dir)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    min_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5

    # substitute with appropriate dataloader
    validation_dataloader = training_datasets

    for epoch in range(start_epoch, start_epoch + max_total_epochs):
        train_loss, train_accuracy = train_epoch(model, training_datasets, criterion, optimizer)
        val_loss = validate_epoch(model, validation_dataloader, criterion)

        if val_loss < min_val_loss:
            save_checkpoint(model, epoch, checkpoints_dir)
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                model, _ = load_latest_available_checkpoint(model, epoch - epochs_no_improve, checkpoints_dir)
                break

        if epoch % save_checkpoint_every_n_epochs == 0:
            save_checkpoint(model, epoch, checkpoints_dir)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}')

    return model
