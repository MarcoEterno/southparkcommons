import os.path

import torch



checkpoints_path = os.path.join(os.path.dirname(os.getcwd()), "model_checkpoints")
logs_path = os.path.join(os.path.dirname(os.getcwd()), "logs", "fit")
data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
custom_data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "custom")
plot_path = os.path.join(os.path.dirname(os.getcwd()), "data", "plots")


#TRAINING SETTINGS
#TODO: implement training parallelism
training_parallelism = True  # if set to True, the training loop will be parallelized using torch.nn.DataParallel. Still to be implemented due to some issues with the model's structure
fast_training = False #False if(torch.has_mps or torch.cuda.is_available()) else True  # if manually set to True, the neural network will lose a layer of depth, but will train faster
num_workers = 0 # number of workers for the dataloader. If set to 0, the dataloader will be single threaded. For some reason, 0 seems to be the optimal value.

#HYPERPARAMETERS
batch_size = 256
learning_rate = 0.001


n_digits_in_number_to_classify = 3  # number of digits to classify
start_epoch = 0  # if set to n, loads the model from checkpoint_{n}.pt
total_epochs_to_train = 15  # total number of epochs that we want to train for
save_checkpoint_every_n_epochs = 1  # save a checkpoint every n epochs


"""
Training times:
30 minutes on m1 macbook air 8gb ram, running on cpu for 3 digits with fast_training=false
20 minutes on m1 macbook air 8gb ram, running on cpu for 3 digits with fast_training=true
7 minutes on m1 macbook air 8gb ram, running on 7 core gpu for 3 digits with fast_training=false 
1.5 minutes on m1 max macbook pro with 32gb ram, running on 32 core gpu for 3 digits with fast_training=false 
"""

"""
recommended traininng times:
- 1 digit: 5 epochs ~5 min on consumer cpu achieves > 99% accuracy
"""

def get_system_device(print_info=False):
    if torch.has_mps:
        if print_info:
            print(f"Using mps device")
        return 'mps'
    elif torch.cuda.is_available():
        if print_info:
            print(f"Using cuda device")
        return 'cuda'
    else:
        if print_info:
            print(f"Using cpu device")
        return 'cpu'

if __name__ == '__main__':
    print(checkpoints_path)