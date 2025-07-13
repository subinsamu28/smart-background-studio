import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from PIL import Image



from u2net import U2NET  # from src/
from data_loader import RescaleT, ToTensorLab  # from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


MODEL_PATH = os.path.join("model", "u2net.pth")

def load_model():
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(MODEL_PATH))
        net.cuda()
    else:
        net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    net.eval()
    return net

from PIL import Image
from torchvision import transforms

def preprocess_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("Input must be a file path or PIL Image.")

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image.size


def postprocess_mask(mask, original_size):
    mask = mask.squeeze().cpu().data.numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).astype(np.uint8)
    return cv2.resize(mask, original_size)

def run_u2net(image_path, output_path="outputs/result.png"):
    net = load_model()
    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1 = net(image_tensor)[0]

    mask = postprocess_mask(d1, original_size)

    original = Image.open(image_path).convert("RGBA")
    original_np = np.array(original)
    alpha = mask.astype(np.uint8)
    rgba = np.dstack((original_np[:, :, :3], alpha))
    Image.fromarray(rgba).save(output_path)

    return output_path


def run_solid_background(image_path, output_path="outputs/solid_result.png", bg_color=(255, 255, 255)):
    net = load_model()
    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1 = net(image_tensor)[0]

    mask = postprocess_mask(d1, original_size)

    # Load original image as RGBA
    original_image = Image.open(image_path).convert("RGBA")
    original_np = np.array(original_image)

    # Create alpha channel from mask
    if mask.shape != original_np.shape[:2]:
        mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # Optional: remove white fringing with erosion
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # Apply mask to alpha channel
    original_np[:, :, 3] = binary_mask

    # Create solid background
    bg = np.full_like(original_np, (*bg_color, 255), dtype=np.uint8)

    # Alpha blend RGBA subject onto solid background
    alpha = binary_mask / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    result = (original_np[:, :, :3] * alpha + bg[:, :, :3] * (1 - alpha)).astype(np.uint8)

    Image.fromarray(result).save(output_path)
    return output_path



def run_background_replace(image_path, bg_image_path, output_path="outputs/replaced_result.png"):
    net = load_model()
    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1 = net(image_tensor)[0]

    mask = postprocess_mask(d1, original_size)

    # Load foreground and background
    original_image = Image.open(image_path).convert("RGB")
    fg = np.array(original_image).astype(np.uint8)
    bg = Image.open(bg_image_path).convert("RGB")
    bg = bg.resize(original_image.size)
    bg = np.array(bg).astype(np.uint8)

    # Apply mask
    mask = np.expand_dims(mask, axis=2) / 255.0
    result = (fg * mask + bg * (1 - mask)).astype(np.uint8)

    Image.fromarray(result).save(output_path)
    return output_path

def run_blur_background(image_path, output_path="outputs/blurred_result.png", blur_intensity=25):
    import cv2
    from PIL import Image
    import numpy as np
    import torch

    net = load_model()
    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1 = net(image_tensor)[0]

    mask = postprocess_mask(d1, original_size)

    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    original = np.array(original_image).astype(np.uint8)

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # Create binary mask
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

    # Create blurred background
    blurred = cv2.GaussianBlur(original, (0, 0), blur_intensity)

    # Extract subject and combine with blurred background
    subject = cv2.bitwise_and(original, mask_3ch)
    inv_mask = cv2.bitwise_not(binary_mask)
    inv_mask_3ch = cv2.merge([inv_mask, inv_mask, inv_mask])
    background = cv2.bitwise_and(blurred, inv_mask_3ch)

    result = cv2.add(subject, background)

    Image.fromarray(result).save(output_path)
    return output_path

def run_smart_resize(image_path, output_path, target_size=(800,800), lock_aspect=True):
    img = Image.open(image_path)
    if lock_aspect:
        img.thumbnail(target_size, Image.ANTIALIAS)
    else:
        img = img.resize(target_size, Image.ANTIALIAS)
    img.save(output_path)
    return output_path

