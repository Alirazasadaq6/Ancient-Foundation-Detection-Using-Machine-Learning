import torch
import segmentation_models_pytorch as smp

# Function to load a trained U-Net model
def load_model(model_path, device=None):
    # Use GPU if available, otherwise use CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create U-Net model with ResNet34 encoder and 1 output class
    model = smp.Unet(
        encoder_name="resnet34",        # Backbone of the model
        encoder_weights="imagenet",     # Use pretrained weights from ImageNet
        in_channels=3,                  # Input has 3 color channels (RGB)
        classes=1                       # Output is a single channel (binary mask)
    )

    # Load the saved model weights from file
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to selected device (CPU or GPU)
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Return the loaded model and device info
    return model, device
