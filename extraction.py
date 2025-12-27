import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
import random
from pathlib import Path
from torchvision import datasets, transforms

# ============================================================================
# 1. DATASET CLASSES
# ============================================================================

class ImageDataset(Dataset):
    """Simple image folder dataset"""
    def __init__(self, folder_path, image_size=256):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) 
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        img = self.transform(img)
        return {"image": img, "path": self.image_files[idx]}

class ImageListDataset(Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


def make_balanced_subset(root, samples_per_class, seed=0):
    random.seed(seed)

    root = Path(root)
    classes = sorted([d for d in root.iterdir() if d.is_dir()])

    selected_paths = []
    selected_labels = []

    for class_idx, class_dir in enumerate(classes):
        images = sorted([p for p in class_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        if len(images) < samples_per_class:
            raise ValueError(
                f"Class {class_dir.name} has only {len(images)} images, "
                f"which is fewer than samples_per_class={samples_per_class}."
            )

        chosen = random.sample(images, samples_per_class)

        for p in chosen:
            selected_paths.append(str(p))
            selected_labels.append(class_idx)

    return selected_paths, selected_labels

# ============================================================================
# 3. CODE EXTRACTION
# ============================================================================

def extract_codes(model, dataloader, device='cuda', structured=True):
    """
    Extract codes from full images (Best for Retain Set).
    Does NOT use NudeNet.
    """
    all_codes = []
    model.eval()
    
    print("Extracting codes from Retain Set (Full Images)...")
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            _, _, info = model.encode(x)
            indices = info[2] 
            
            if structured:
                # Reshape to [B, H, W]
                if indices.dim() == 2:
                    B = indices.shape[0]
                    HW = indices.shape[1]
                    H = W = int(np.sqrt(HW))
                    indices = indices.reshape(B, H, W)
                elif indices.dim() == 4:
                    indices = indices.squeeze(1)

            all_codes.append(indices.cpu())
    
    return torch.cat(all_codes, dim=0)

def create_mask(detections, img_size=(256, 256), expand=0):
    """
    Create a binary mask for any provided detections.
    
    Unlike the previous version, this does NOT filter by class or score.
    It assumes the input 'detections' list has already been filtered 
    by the parent function (extract_explicit_codes).
    """
    # Initialize a black mask (0.0)
    mask = torch.zeros(img_size, dtype=torch.float32)
    W, H = img_size
    
    for det in detections:
        
        # NudeNet boxes are [x_top_left, y_top_left, width, height]
        x, y, w, h = det["box"]
        
        # Calculate expansion (padding around the object)
        dx, dy = int(w * expand), int(h * expand)
        
        # Calculate coordinates with boundary checks
        # We cast to int() because NudeNet sometimes returns floats
        x1 = max(0, int(x - dx))
        y1 = max(0, int(y - dy))
        x2 = min(W, int(x + w + dx))
        y2 = min(H, int(y + h + dy))
        
        # Set the region to 1.0 (white)
        # Note: PyTorch tensors use [Height, Width] indexing (y, x)
        mask[y1:y2, x1:x2] = 1.0
    
    return mask


def extract_codes_nudenet(
    model,
    detector,
    dataloader,
    target_classes,
    device='cuda',
    expand=0.0,
    min_score=0.5,
    mask_breasts=False
):
    """
    Extract explicit VQ codes using NudeNet detections (masked strategy only).
    """
    print(f"\nEXTRACTING CODES (Masked strategy, Occlusion: {mask_breasts})...")
    
    all_codes = []
    all_masks = []
    total_items = 0
    
    model.eval()
    vqgan_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 1. Get images
            x = batch["image"].to(device)

            # 2. Denormalize to [0, 1] for detection
            imgs_denorm = (x * 0.5 + 0.5).clamp(0, 1)
            pil_imgs = [T.ToPILImage()(img.cpu()) for img in imgs_denorm]

            for pil_img in pil_imgs:
                # Convert PIL → NumPy (BGR) for NudeNet
                img_numpy = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

                detections = detector.detect(img_bgr)

                valid_dets = [
                    d for d in detections
                    if d['score'] >= min_score and d['class'] in target_classes
                ]

                if not valid_dets:
                    continue

                # Optional occlusion
                if mask_breasts:
                    draw = ImageDraw.Draw(pil_img)
                    W, H = pil_img.size
                    for d in valid_dets:
                        x, y, w, h = d['box']
                        dx, dy = int(w * expand), int(h * expand)
                        x1, y1 = max(0, x - dx), max(0, y - dy)
                        x2, y2 = min(W, x + w + dx), min(H, y + h + dy)
                        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

                # Create spatial mask
                mask = create_mask(valid_dets, expand=expand)
                if mask.sum() == 0:
                    continue

                # Encode full image
                x_full = vqgan_transform(pil_img).unsqueeze(0).to(device)
                _, _, info = model.encode(x_full)
                indices = info[2]

                if indices.dim() == 2:
                    h = w = int(np.sqrt(indices.shape[1]))
                    indices = indices.reshape(1, h, w)
                elif indices.dim() == 4:
                    indices = indices.squeeze(0).squeeze(0).unsqueeze(0)

                H_lat, W_lat = indices.shape[1], indices.shape[2]
                mask_lat = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H_lat, W_lat),
                    mode='nearest'
                ).squeeze()

                all_codes.append(indices.cpu())
                all_masks.append(mask_lat.cpu())
                total_items += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches ({total_items} items)...")

    if not all_codes:
        raise ValueError("No explicit content detected. Check threshold/images.")

    codes_out = torch.cat(all_codes, dim=0)
    masks_out = torch.stack(all_masks, dim=0)

    print(f"Done. Extracted {total_items} items.")
    return codes_out, masks_out