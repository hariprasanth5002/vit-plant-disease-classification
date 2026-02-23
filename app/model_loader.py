import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path, num_classes):

    # Load architecture WITHOUT pretrained weights
    model = models.vit_b_16(weights=None)

    # Replace classifier head (same as training)
    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        num_classes
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()

    return model
