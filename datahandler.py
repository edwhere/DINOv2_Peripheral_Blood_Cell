
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 256  # Size of the input image (images are resized to this value)
NUM_WORKERS = 4   # Number of workers for data loading

# Training transforms
def get_trn_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_val_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform


def get_datasets(trn_dir_path, val_dir_path):
    """ Create handlers for the train and validation datasets. Return the TRN and VAL dataset handlers
    and the associated classes.
    """
    dataset_train = datasets.ImageFolder(trn_dir_path, transform=(get_trn_transform(IMAGE_SIZE)))
    dataset_valid = datasets.ImageFolder(val_dir_path, transform=(get_val_transform(IMAGE_SIZE)))
    return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """ Return the loader functions for TRN and VAL datasets. The function uses the dataset handlers
    obtained using the get_datasets() function.
    """
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, valid_loader