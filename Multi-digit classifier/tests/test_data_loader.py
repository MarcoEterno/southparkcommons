import unittest

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from source.data_loader_old import get_MNIST_data, check_bounds_for_n_digits_dataset, get_n_digits_dataset, augment_data, \
    get_MNIST_train_test_data


class DataLoaderTest(unittest.TestCase):
    def test_get_MNIST_data_length(self):
        result = get_MNIST_data(train=True, data_loading=False)
        self.assertEqual(len(result), 60000)
        result = get_MNIST_data(train=False, data_loading=False)
        self.assertEqual(len(result), 10000)

    def test_check_bounds_for_single_digit_dataset(self):
        with self.assertRaises(ValueError):
            check_bounds_for_n_digits_dataset(0)

    def test_check_bounds_for_double_digit_dataset(self):
        with self.assertRaises(ValueError):
            check_bounds_for_n_digits_dataset(10)

    def test_n_digits_dataset_image_dimension(self):
        result = get_n_digits_dataset(2)
        for batch in iter(result):
            for img, label in zip(*batch):
                self.assertEqual(img.shape, (1, 28, 28 * 2))

    def test_get_MNIST_data_type(self):
        result = get_MNIST_data(train=True, data_loading=False)
        self.assertIsInstance(result, datasets.MNIST)

    def test_get_MNIST_data_transform(self):
        result = get_MNIST_data(train=True, data_loading=False)
        self.assertIsInstance(result.transform, transforms.ToTensor)

    def test_get_n_digits_dataset_type(self):
        result = get_n_digits_dataset(2)
        self.assertIsInstance(result, DataLoader)

    def test_augment_data_length(self):
        original_dataset = get_MNIST_data(train=True, data_loading=True)
        mini_dataset = list(original_dataset)[:10]
        augmented_dataset = augment_data(mini_dataset, image_upsizing=2)
        self.assertEqual(len(augmented_dataset), len(mini_dataset))

    def test_augment_data_type(self):
        original_dataset = get_MNIST_data(train=True, data_loading=True)
        mini_dataset = list(original_dataset)[:10]
        augmented_dataset = augment_data(mini_dataset, image_upsizing=2)
        self.assertIsInstance(augmented_dataset, DataLoader)

    def test_get_MNIST_train_test_data_length(self):
        train_data, test_data = get_MNIST_train_test_data(data_loading=False)
        self.assertEqual(len(train_data), 48000)
        self.assertEqual(len(test_data), 12000)

if __name__ == '__main__':
    unittest.main()
