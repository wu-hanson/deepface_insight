import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet_model(num_classes: int = 2, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, num_classes)
    

    return model


def load_resnet_model(path: str, device: torch.device, num_classes: int = 2):

    model = get_resnet_model(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model