import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from PIL import Image

from source.config import get_system_device, batch_size, data_path, custom_data_path, num_workers


def get_MNIST_data(train=True, data_loading=True, print_info=True):
    """
    Loads the MNIST dataset
    :param train: if True, loads the training dataset, otherwise loads the validation dataset
    :param data_loading: if True, returns the dataset as a DataLoader object, otherwise returns it as a Dataset object
    :param print_info: if True, prints information about the process being executed
    """
    if print_info:
        print(f"Loading the MNIST {'training' if train else 'validation'} dataset")
    data = datasets.MNIST(
        root=data_path,
        train=train,
        download=True,
        transform=ToTensor()
    )
    if data_loading:
        dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
        return dataset
    else:
        return data


def train_test_split(dataset: Dataset, train_size=0.8, data_loading=True):
    """
    Splits a dataset into a train and test dataset
    :param dataset: the dataset to split
    :param train_size: the size of the train dataset as a fraction of the original dataset
    :param data_loading: if True, returns the train and test datasets as DataLoader objects,
    otherwise returns them as Dataset objects
    :return: the train and test datasets
    """
    n_train_samples = int(len(dataset) * train_size)
    n_test_samples = len(dataset) - n_train_samples
    train_data, test_data = torch.utils.data.random_split(dataset, [n_train_samples, n_test_samples])
    if data_loading:
        train_data, test_data = torch.utils.data.random_split(dataset, [n_train_samples, n_test_samples])
        train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataset, test_dataset
    else:
        return train_data, test_data


def get_MNIST_train_test_data(train_size=0.8, data_loading=True, print_info=True):
    """
    Loads the MNIST training and test datasets
    :param train_size: the size of the train dataset as a fraction of the original dataset
    :param data_loading: if True, returns the train and test datasets as DataLoader objects, otherwise returns them as Dataset objects
    """
    if print_info:
        print(f"Loading the MNIST training dataset")
    original_training_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=ToTensor()
    )
    # Performing a train test split on the MNIST dataset
    train_data, test_data = train_test_split(original_training_data, train_size=train_size, data_loading=data_loading)
    return train_data, test_data


def check_bounds_for_n_digits_dataset(n: int):
    """
    Checks that the input is a valid number of digits for the dataset
    :param n: the number of digits to check
    """
    if n < 1:
        raise ValueError("Expected number of digits for the dataset to create is greater than 1")
    if n > 9:
        raise ValueError("Expected number of digits for the dataset to create is less than 10.")


def generate_n_digits_from_dataset(n: int, dataset: Dataset, data_loading=True, augment_data=False, scale_data_linearly=False,
                                   device=get_system_device()):
    """
    Given a dataset, returns a dataset where each data is composed of n images from the original dataset
    combined side by side
    :param n: number of digits to combine into a single image
    :param dataset: the dataset to generate the images from
    :param data_loading: if True, returns the dataset as a DataLoader object, otherwise returns it as a Dataset object
    :param augment_data: if True, augments the dataset by rotating and cropping the images
    :param scale_data_linearly: if True, produces a dataset of size linear with the number of digits to recognize
    :param device: the device to load the dataset on
    """
    check_bounds_for_n_digits_dataset(n)

    if n==1:
        if data_loading:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            return dataset

    tuples = []
    # loop through the dataset and combine n images into a single image
    for idx in range(len(dataset)):
        for m in range(n if scale_data_linearly else 1):
            for i in range(n):
                img, label = dataset[(idx + (i + n * i)*(m+1)) % len(dataset)] # you can cleverly demonstrate that with
                # this rule, the same two digits can never appear in two different images, thus reducing overfitting
                img = transforms.ToPILImage()(img)
                #combine the images side by side
                if i == 0:
                    combined_img = Image.new('L', (img.width * n, img.height))
                    combined_img.paste(img, (0, 0))
                    combined_label = label
                else:
                    combined_img.paste(img, (img.width * i, 0))
                    combined_label = combined_label * 10 + label

            # convert images to tensor
            combined_img = transforms.ToTensor()(combined_img).to(device)
            combined_label = torch.tensor(combined_label, dtype=torch.long).to(device)

            tuples.append((combined_img, combined_label))

    if data_loading:
        return DataLoader(tuples, batch_size=batch_size, shuffle=True)
    else:
        return tuples

def get_n_digits_train_validation_test_dataset(n: int, augment_data=False, scale_data_linearly=False,
                                               device=get_system_device()):
    """
        Creates a dataset containing images of n digits from MNIST by combining n images from the MNIST dataset side by side
        :param n: number of digits to combine into a single image
        :param train: if True, returns the training dataset, otherwise returns the test dataset
        :param augment_data: if True, augments the dataset by rotating and cropping the images
        :param scale_data_linearly: if True, produces a dataset of size linear with the number of digits to recognize
        :param device: the device to load the dataset on
        :return: train,validation and test dataloaders containing images of n digits from MNIST
    """
    check_bounds_for_n_digits_dataset(n)

    # creating the paths for the datasets
    train_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_train.pt')
    validation_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_validation.pt')
    test_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_test.pt')

    # checking if the datasets already exist and load them if they do
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path) and os.path.exists(
            validation_dataset_path):
        print(f"Loading the {n} digits train, validation and test datasets from {custom_data_path}")
        train_dataset = DataLoader(torch.load(train_dataset_path, map_location='cpu'),num_workers=num_workers, batch_size=batch_size, shuffle=True)
        validation_dataset = DataLoader(torch.load(validation_dataset_path, map_location='cpu'), num_workers=num_workers, batch_size=batch_size, shuffle=True)
        test_dataset = DataLoader(torch.load(test_dataset_path, map_location='cpu'), num_workers=num_workers, batch_size=batch_size, shuffle=True)
        return train_dataset, validation_dataset, test_dataset
    else:
        # if the datasets do not exist, create them
        train_data, test_data = get_MNIST_train_test_data(data_loading=False)
        validation_data = get_MNIST_data(train=False, data_loading=False)

        print(f"Creating the {n} digits train dataset from the MNIST train dataset. This might take a while")
        train_dataset = generate_n_digits_from_dataset(n, train_data, data_loading=False, augment_data=augment_data, scale_data_linearly=scale_data_linearly)
        print(f"Created the {n} digits train dataset. Creating the {n} digits test and validation datasets")
        test_dataset = generate_n_digits_from_dataset(n, test_data, data_loading=False, augment_data=augment_data, scale_data_linearly=scale_data_linearly)
        validation_dataset = generate_n_digits_from_dataset(n, validation_data, data_loading=False, augment_data=augment_data, scale_data_linearly=scale_data_linearly)

        print(f"Created the {n} digits test and validation datasets. Saving the {n} digits train, validation and test datasets to {custom_data_path}")
        torch.save(train_dataset, train_dataset_path)
        torch.save(test_dataset, test_dataset_path)
        torch.save(validation_dataset, validation_dataset_path)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, validation_dataloader, test_dataloader
