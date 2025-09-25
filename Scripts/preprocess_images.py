import os
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np

def load_resnet50(device='cuda'):
    """Load pretrained ResNet50 without the final classification layer."""
    model = resnet50(weights='IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc layer
    model.eval()
    model.to(device)
    return model

def preprocess_image(img_path):
    """Open image, resize, normalize and convert to tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)  # add batch dimension

def extract_features_for_folder(img_dir, out_dir, device='cuda'):
    """
    Extract features for all images in img_dir and save them as .npy files in out_dir.
    Each image becomes a 2048-dim vector from ResNet50 pooled layer.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = load_resnet50(device)
    
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        try:
            x = preprocess_image(img_path).to(device)
            with torch.no_grad():
                feat = model(x).squeeze().cpu().numpy()
            np.save(os.path.join(out_dir, img_name.replace('.jpg', '.npy')), feat)
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

def preprocess_images(img_dir, out_dir, device='cuda'):
    """
    Full pipeline: extract features and save locally in Colab storage.
    Returns a dict with info about saved features.
    """
    extract_features_for_folder(img_dir, out_dir, device)
    num_images = len(os.listdir(out_dir))
    return {"local_out_dir": out_dir, "num_images": num_images}