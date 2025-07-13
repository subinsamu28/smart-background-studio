import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from u2net import U2NET  # model definition
from data_loader import RescaleT, ToTensorLab

# Paths
MODEL_PATH = os.path.join("model", "u2net.pth")
INPUT_IMAGE_PATH = os.path.join("images", "test.jpg")
OUTPUT_IMAGE_PATH = os.path.join("outputs", "result.png")

# Preprocess image for the model
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

# Postprocess the output mask
def postprocess_mask(mask, original_size):
    mask = mask.squeeze().cpu().data.numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).astype(np.uint8)
    return cv2.resize(mask, original_size)

def save_output(original_image, mask, output_path):
    # Convert original image to RGBA
    original = original_image.convert("RGBA")
    original_np = np.array(original)

    # Resize mask if needed
    if mask.shape != original_np.shape[:2]:
        mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))

    # Normalize mask to [0, 255] and apply as alpha channel
    alpha = (mask).astype(np.uint8)

    # Combine RGB with alpha
    rgba = np.dstack((original_np[:, :, :3], alpha))

    # Save result
    Image.fromarray(rgba).save(output_path)
    print(f"✅ Transparent image saved to: {output_path}")



def main():
    print("🔄 Loading model...")
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(MODEL_PATH))
        net.cuda()
    else:
        net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    net.eval()

    print("🖼️ Processing image...")
    image_tensor, original_image = preprocess_image(INPUT_IMAGE_PATH)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1 = net(image_tensor)[0]

    mask = postprocess_mask(d1, original_image.size)
    save_output(original_image, mask, OUTPUT_IMAGE_PATH)

if __name__ == "__main__":
    main()
