import unittest
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from source.data_loader import get_MNIST_data, train_test_split, get_MNIST_train_test_data, check_bounds_for_n_digits_dataset, generate_n_digits_from_dataset, get_n_digits_train_validation_test_dataset

class DataLoaderScratchTest(unittest.TestCase):
    def test_get_MNIST_data(self):
        # Test the get_MNIST_data function
        data = get_MNIST_data()
        self.assertIsInstance(data, DataLoader)

    def test_train_test_split(self):
        # Test the train_test_split function
        original_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
        train_data, test_data = train_test_split(original_data)
        self.assertIsInstance(train_data, DataLoader)
        self.assertIsInstance(test_data, DataLoader)

    def test_get_MNIST_train_test_data(self):
        # Test the get_MNIST_train_test_data function
        train_data, test_data = get_MNIST_train_test_data()
        self.assertIsInstance(train_data, DataLoader)
        self.assertIsInstance(test_data, DataLoader)

    def test_check_bounds_for_n_digits_dataset(self):
        # Test the check_bounds_for_n_digits_dataset function
        with self.assertRaises(ValueError):
            check_bounds_for_n_digits_dataset(0)
        with self.assertRaises(ValueError):
            check_bounds_for_n_digits_dataset(10)
        self.assertIsNone(check_bounds_for_n_digits_dataset(5))

    def test_generate_n_digits_from_dataset(self):
        # Test the generate_n_digits_from_dataset function
        original_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
        generated_data = generate_n_digits_from_dataset(2, original_data)
        self.assertIsInstance(generated_data, DataLoader)

    def test_get_n_digits_train_validation_test_dataset(self):
        # Test the get_n_digits_train_validation_test_dataset function
        train_data, validation_data, test_data = get_n_digits_train_validation_test_dataset(2)
        self.assertIsInstance(train_data, DataLoader)
        self.assertIsInstance(validation_data, DataLoader)
        self.assertIsInstance(test_data, DataLoader)

if __name__ == '__main__':
    unittest.main()