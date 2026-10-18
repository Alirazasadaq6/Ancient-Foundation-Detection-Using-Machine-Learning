import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn

# Function to load a trained DeepLabV3 model
def load_model(model_path, device=None):
    # Use GPU if available, otherwise use CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model architecture (no pretrained weights, no auxiliary loss)
    model = deeplabv3_resnet50(weights=None, aux_loss=False)

    # Change the output layer to 1 channel for binary segmentation
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    # Load the trained model weights from file (ignore extra keys if any)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # Move model to selected device (CPU or GPU)
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Return the model and device
    return model, device