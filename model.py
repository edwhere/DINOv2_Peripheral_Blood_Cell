"""Implement a DINOv2 model with a different header."""

import torch
from collections import OrderedDict

import constants as con
import errordefs as err


def build_model(num_classes: int=2, fine_tune: bool=False):
    backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    model = torch.nn.Sequential(OrderedDict([
        ('backbone', backbone_model),
        ('head', torch.nn.Linear(
            in_features=384, out_features=num_classes, bias=True
        ))
    ]))

    if not fine_tune:
        for params in model.backbone.parameters():
            params.requires_grad = False

    return model


def load_stored_data(file_path: str):
    """Load the model from file_path, initialize the model architecture and assign stored weights.
    Return:
        The instantiated model and the list of labels.
    """

    # Load stored model. This operation returns a dictionary.
    model_data = torch.load(file_path, weights_only=False)

    if 'storage_type' not in model_data.keys():
        raise err.UnknownStorageFormat("Incompatible stored model format.")

    if model_data['storage_type'] not in con.FORMAT_TYPES:
        raise err.UnknownStorageFormat("Incompatible stored model format.")

    if 'labels' not in model_data.keys():
        raise err.UnknownStorageFormat("Incompatible stored model format.")

    labels = model_data["labels"]
    num_classes = len(labels)

    # Instantiate model architecture
    backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    model = torch.nn.Sequential(OrderedDict([
        ('backbone', backbone_model),
        ('head', torch.nn.Linear(
            in_features=384, out_features=num_classes, bias=True
        ))
    ]))

    # Instantiate model weights
    model.load_state_dict(state_dict=model_data['model_state_dict'])

    return model, labels



if __name__ == "__main__":
    test1 = False
    test2 = True

    if test1:
        # Test script for model creation.
        my_model = build_model()
        print(my_model)
        print('---------------------------------------------------------------')

        total_params = sum(p.numel() for p in my_model.parameters())
        print(f"total parameters: {total_params}")

        trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
        print(f"trainable parameters: {trainable_params}")

    elif test2:
        file_loc = "/home/xuser/Experzone/Dino32/best_dv2model.pth"
        loaded_model, loaded_labels = load_stored_data(file_loc)
        print()
        print("loaded labels:", loaded_labels)

        total_params = sum(p.numel() for p in loaded_model.parameters())
        print(f"total parameters: {total_params}")
