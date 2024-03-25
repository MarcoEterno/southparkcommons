import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from PIL import Image

from source.config import get_system_device, batch_size, data_path, custom_data_path


def get_MNIST_data(train=True, data_loading=True):
    """
    Loads the MNIST dataset
    """
    print(f"Loading the MNIST {'training' if train else 'test'} dataset")
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
    :param data_loading: if True, returns the train and test datasets as DataLoader objects, otherwise returns them as Dataset objects
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


def get_MNIST_train_test_data(train_size = 0.8, data_loading=True):
    """
    Loads the MNIST training and test datasets
    :param data_loading: if True, returns the train and test datasets as DataLoader objects, otherwise returns them as Dataset objects
    """
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
    if n < 1:
        raise ValueError("Expected number of digits for the dataset to create is greater than 1")
    if n > 9:
        raise ValueError("Expected number of digits for the dataset to create is less than 10.")


# create a dataset containing images of n digits from MNIST
def get_n_digits_dataset(n: int, train=True, augment_data=False, scale_data_linearly=False, device=get_system_device()):
    """
    Creates a dataset containing images of n digits from MNIST by combining n images from the MNIST dataset side by side
    :param n: number of digits to combine into a single image
    :param train: if True, returns the training dataset, otherwise returns the test dataset
    :param augment_data: if True, augments the dataset by rotating and cropping the images
    :param scale_data_linearly: if True, produces a dataset of size linear with the number of digits to recognize
    :param device: the device to load the dataset on
    """
    # check edge cases for n_digits_dataset
    check_bounds_for_n_digits_dataset(n)
    if n == 1:
        return get_MNIST_data(train=train, data_loading=True)

    dataset_type = 'train' if train else 'test'
    dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_{dataset_type}.pt')
    # if n digits dataset already exists, load it
    if os.path.exists(dataset_path):
        print(f"Loading the {n} digits {dataset_type} dataset")
        n_digits_dataset = DataLoader(torch.load(dataset_path, map_location='cpu'), batch_size=batch_size, shuffle=True)
        return n_digits_dataset

    # if n digits dataset does not exist, create it
    tuples = []
    mnist_dataset = get_MNIST_data(train=train, data_loading=False)
    print(f"Creating the {n} digits {dataset_type} dataset")

    for idx in range(len(mnist_dataset)):
        # Get n consecutive images (wrapping around at the end)
        for i in range(n):
            img, label = mnist_dataset[(idx + i) % len(mnist_dataset)]
            img = transforms.ToPILImage()(img)
            # Combine images side by side
            if i == 0:
                combined_img = Image.new('L', (28 * n, 28))
                combined_img.paste(img, (0, 0))
                combined_label = label
            else:
                combined_img.paste(img, (28 * i, 0))
                combined_label = 10 * combined_label + label

        # Convert combined image and label back to tensor
        combined_img = transforms.ToTensor()(combined_img).to(device)
        combined_label = torch.tensor(combined_label, dtype=torch.long).to(device)

        tuples.append((combined_img, combined_label))

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"{n} digits {dataset_type} dataset created. Saving dataset to {dataset_path}")
    torch.save(tuples, dataset_path)
    print("Loading the dataset")
    n_digits_dataset = DataLoader(tuples, batch_size=batch_size, shuffle=True)
    return n_digits_dataset


def create_n_digits_train_test_dataset(n: int, train_dataset_path, test_dataset_path, augment_data=False, scale_data_linearly=False,
                            device=get_system_device()):
    original_train_dataset = get_n_digits_dataset(n, train=True, augment_data=augment_data,
                                                  scale_data_linearly=scale_data_linearly, device=device)
    original_validation_dataset = get_n_digits_dataset(n, train=False, augment_data=augment_data,
                                                       scale_data_linearly=scale_data_linearly, device=device)
    train_dataset, test_dataset = train_test_split(original_train_dataset, train_size=0.8, data_loading=True)
    os.makedirs(os.path.dirname(train_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_dataset_path), exist_ok=True)
    print(
        f"{n} digits train and test dataset created. Saving dataset for future use to: {train_dataset_path} and {test_dataset_path}")
    torch.save(train_dataset, train_dataset_path)
    torch.save(test_dataset, test_dataset_path)
    return train_dataset, test_dataset


def get_n_digits_train_validation_test_datasets(n: int, augment_data=False, scale_data_linearly=False, device=get_system_device()):
    """
    Creates a dataset containing images of n digits from MNIST by combining n images from the MNIST dataset side by side
    :param n: number of digits to combine into a single image
    :param train: if True, returns the training dataset, otherwise returns the test dataset
    :param augment_data: if True, augments the dataset by rotating and cropping the images
    :param scale_data_linearly: if True, produces a dataset of size linear with the number of digits to recognize
    :param device: the device to load the dataset on
    """

    train_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_train.pt')
    test_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_test.pt')
    validation_dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_validation.pt')

    # if n digits train and test datasets already exists, load it
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        print(f"Loading the {n} digits train and test dataset")
        n_digits_train_dataset = DataLoader(torch.load(train_dataset_path, map_location='cpu'), batch_size=batch_size, shuffle=True)
        n_digits_test_dataset = DataLoader(torch.load(test_dataset_path, map_location='cpu'), batch_size=batch_size, shuffle=True)
    else:
        # if n digits dataset does not exist, create it
        n_digits_train_dataset, n_digits_test_dataset = create_n_digits_train_test_dataset(n, train_dataset_path, test_dataset_path,
                                      augment_data=augment_data, scale_data_linearly=scale_data_linearly)
    n_digits_validation_dataset = get_n_digits_dataset(n, train=False, augment_data=augment_data, scale_data_linearly=scale_data_linearly)
    return n_digits_train_dataset, n_digits_validation_dataset, n_digits_test_dataset



def augment_data(dataset: DataLoader, image_upsizing=1, batch_size=batch_size, device=get_system_device()):
    """
    Augments the dataset by rotating, translating and cropping the images
    :param dataset: the dataset to augment
    :param image_upsizing: the factor by which to scale up the dimension of the images in the dataset
    :param batch_size: the batch size of the dataset
    :param device: the device to load the dataset on
    """
    augmented_dataset = []
    for batch_images, batch_labels in dataset:
        for img, label in zip(batch_images, batch_labels):
            # TODO: add resizing and translation
            # if image_upsizing > 1:
            # img = transforms.Resize((28 * image_upsizing, 28 * image_upsizing))(img)

            img = transforms.ToPILImage()(img)
            img = transforms.RandomRotation(20)(img)
            img = transforms.RandomResizedCrop(224)(img)
            img = transforms.ToTensor()(img).to(device)
            augmented_dataset.append((img, label))
    return DataLoader(augmented_dataset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    pass
