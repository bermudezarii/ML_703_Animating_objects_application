"""
    Definition of Models

    * Implemented Models:
        - ResNet 18
        - EfficientNet B4
"""

from torch import nn
from torchvision import models
from .hyperparameters import parameters as params


def resnet18():
    """
        efficientnet ResNet 18 model definition.
    """
    out_features = params['out_features']

    model = models.resnet18(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features)

    return model


def efficientnet():
    """
        efficientnet EfficientNet B4 model definition.
    """
    out_features = params['out_features']

    model = models.efficientnet_b4(pretrained=True)

    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=out_features)

    return model
