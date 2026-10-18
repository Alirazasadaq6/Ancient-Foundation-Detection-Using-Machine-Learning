import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.interpolate import splprep, splev

# Function to smooth the shape of a contour
def smooth_contour(contour, smoothing=0.003):
    # Get contour points
    pts = contour[:, 0, :]
    if not np.array_equal(pts[0], pts[-1]):
        # Make it a closed loop if it's not
        pts = np.vstack([pts, pts[0]])
    # Spline interpolation
    tck, u = splprep([pts[:,0], pts[:,1]], s=smoothing, per=True)
    # More points for smoother curve
    u_new = np.linspace(0, 1, len(pts)*15)
    x_new, y_new = splev(u_new, tck)
    smooth_pts = np.array([x_new, y_new]).T.astype(np.int32)
    return smooth_pts

# Function to run detection and draw smooth lines on objects
def run_detection(pil_image, model, device, threshold=0.5):
    model.eval()  # Set model to evaluation mode

    # Convert PIL image to tensor and add batch dimension
    transform = transforms.Compose([transforms.ToTensor()])
    image_input = transform(pil_image).unsqueeze(0).to(device)

    # Predict the segmentation mask
    with torch.no_grad():
        output = model(image_input)
        # If model returns dictionary (like DeepLab), use the 'out' key
        if isinstance(output, dict) and 'out' in output:
            output = output['out']

        # Convert output to probability mask and binary mask
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask_bin = (pred_mask > threshold).astype(np.uint8)

    # Convert image and mask to numpy format
    image_np = np.array(pil_image)
    mask_uint8 = pred_mask_bin * 255  # Convert to 0–255 range

    # Remove small noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Dilate the mask to get edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    dilated_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=1)

    # Get edges by subtracting clean mask from dilated mask
    edge_mask = cv2.subtract(dilated_mask, clean_mask)

    # Find contours from the edge mask
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Copy the original image to draw on it
    image_with_lines = image_np.copy()

    # Draw smooth contour lines on the image
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue  # Skip very small areas
        smooth_pts = smooth_contour(cnt)
        cv2.polylines(image_with_lines, [smooth_pts], isClosed=True, color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)

    # Return the final image with detected lines
    return Image.fromarray(image_with_lines)