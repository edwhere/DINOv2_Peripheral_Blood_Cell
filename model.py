"""Implement a DINOv2 model with a different header."""

import torch

from collections import OrderedDict

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

if __name__ == "__main__":
    # Test script for model creation.
    my_model = build_model()
    print(my_model)
    print('---------------------------------------------------------------')

    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable_params}")
