from config import get_system_device
from image_classifier import ImageClassifier


def test_model_performance(clf: ImageClassifier, test_datasets, device=get_system_device()):
    """
    Tests the model performance on the test dataset
    :param clf: the model to test
    :param test_datasets: the test dataset
    :param device: the device to use for inference
    """
    clf.eval()
    tot_accuracy = 0
    for x,y in test_datasets:
        x = x.to(device)
        y = y.to(device)
        output = clf(x)
        accuracy = (output.argmax(1) == y).sum().item() / len(y)
        tot_accuracy += accuracy
    tot_accuracy = tot_accuracy / len(test_datasets)
    print("model accuracy on test dataset is: ", tot_accuracy)
    return tot_accuracy
