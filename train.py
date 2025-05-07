"""The main script for training a DINO V2 classification model."""
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm

import model as mod
import utils as utl
import constants as con
import datahandler as dhr

# Defined constants for this script
SEED = 17
EPOCHS = 10
LRATE = 0.0001
BATCH = 32
KEYWORD = "dv2model"
SCHEDULER = [1000]

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class DirectoryNotFoundError(Exception):
    """Raise an exception if a directory does not exist."""
    pass


class InputData:
    def __init__(self, root_path: str):
        self.__root = root_path

    @property
    def root_path(self):
        return self.__root

    @property
    def trn_path(self):
        return os.path.join(self.__root, con.TRN_DIR_NAME)

    @property
    def val_path(self):
        return os.path.join(self.__root, con.VAL_DIR_NAME)


def parse_arguments():
    """Parse command-line arguments and return the arguments object."""
    parser = argparse.ArgumentParser(description="Train a DINOv2 classifier.")
    required = parser.add_argument_group("requirements")

    required.add_argument("--data_dir_path", type=str, required=True,
                          help="Path to data directory containing TRN and VAL folders.")

    required.add_argument("--results_dir_path", type=str, required=True,
                          help="Path to a directory that will contain the results.")

    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (integer). Default: {EPOCHS}")

    parser.add_argument("--lrate", type=float, default=LRATE,
                        help=f"Learning rate (real number between 0 and 1. Default: {LRATE}")

    parser.add_argument("--batch", type=int, default=BATCH,
                        help=f"Batch size (integer). Default: {BATCH}")

    parser.add_argument("--keyword", type=str, default=KEYWORD,
                        help=f"A keyword included in the file name of the saved model. Default: {KEYWORD}")

    parser.add_argument("--fine_tune", action='store_true',
                        help="Flag used to fine-tune the model instead of simply doing transfer learning.")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir_path):
        raise DirectoryNotFoundError(f"Unavailable data directory: {args.data_dir_path}")

    if args.lrate <= 0 or args.lrate >= 1:
        raise ValueError("Learning rate must be larger than 0 and less than 1.")

    return args


# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def get_device():
    """Determine if a machine runs on a gpu or cpu. Includes M1/M2/M3 gpus. Return the device
    identifier for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


def main():
    """Main sequence of operations."""
    args = parse_arguments()

    # Create results directory
    os.makedirs(args.results_dir_path, exist_ok=True)

    # Determine target device
    target_device = get_device()

    # Get dataset handlers and data loaders
    input_data = InputData(root_path=args.data_dir_path)
    trn_data_handler, val_data_handler, labels = dhr.get_datasets(trn_dir_path=input_data.trn_path,
                                                                  val_dir_path=input_data.val_path)

    trn_loader, val_loader = dhr.get_data_loaders(dataset_train=trn_data_handler,
                                                  dataset_valid=val_data_handler,
                                                  batch_size=args.batch)

    # Create a model representation, load parameters, and store it in the selected device
    model = mod.build_model(num_classes=len(labels), fine_tune=args.fine_tune).to(target_device)

    # Define operational functions for training/validation
    optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=args.lrate)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=SCHEDULER, gamma=0.1)

    # Initialize `SaveBestModel` class.
    save_best_model = utl.SaveBestModel()

    # Lists to keep track of losses and accuracies
    trn_loss, val_loss, trn_acc, val_acc = list(), list(), list(), list()

    # Start the training procedure
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} of {args.epochs}")
        trn_epoch_loss, trn_epoch_acc = train(model, trn_loader, optimizer, criterion, target_device)
        val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, target_device)

        trn_loss.append(trn_epoch_loss)
        val_loss.append(val_epoch_loss)
        trn_acc.append(trn_epoch_acc)
        val_acc.append(val_epoch_acc)

        print(f"trn loss: {trn_epoch_loss:.3f}, trn acc: {trn_epoch_acc:.3f}, "
              f"val loss: {val_epoch_loss:.3f}, val acc: {val_epoch_acc:.3f}")

        save_best_model(val_epoch_loss, epoch, model, args.results_dir_path, args.keyword)
        print('-' * 50)
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        print(f"LR for next epoch: {last_lr}")

    # Save the trained model weights.
    utl.save_model(args.epochs, model, optimizer, criterion, args.results_dir_path, args.keyword)

    # Save the loss and accuracy plots.
    utl.save_plots(trn_acc, val_acc, trn_loss, val_loss, args.results_dir_path)
    print('Finished training a DINOv2 model')

if __name__ == "__main__":
    main()
