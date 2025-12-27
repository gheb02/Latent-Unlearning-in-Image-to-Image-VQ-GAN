import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
import random
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from Nudenet.nudenet import nudenet



class LogisticEmbeddingClassifier(nn.Module):
    def __init__(self, codebook_size, emb_dim=32, dropout = 0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(codebook_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        
        self.classifier = nn.Linear(32 * 32 * emb_dim, 1)
        
    def forward(self, codes):
        """
        codes: [B, 32, 32] con dtype torch.long
        """
        x = self.embedding(codes)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        
        logits = self.classifier(x)
        
        return logits.squeeze(1) 


class CodeTensorDataset(Dataset):
    def __init__(self, tensor_3d, label):
        """
        tensor_3d: shape [N, 32, 32]
        label: int (0 o 1)
        """
        self.tensor = tensor_3d
        self.label = label
    
    def __len__(self):
        return self.tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            "codes": self.tensor[idx].long(),               
            "label": torch.tensor(self.label, dtype=torch.long)
        }

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

def collect_image_paths(class_dirs):
    """
    class_dirs: list of folder paths, each containing images
    returns: list of image file paths
    """

    class_dirs = [Path(d) for d in class_dirs]
    all_paths = []

    for class_dir in class_dirs:
        if not class_dir.is_dir():
            raise ValueError(f"{class_dir} is not a directory")

        images = sorted([
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        all_paths.extend(str(p) for p in images)

    return all_paths


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

def reconstruct_and_compare(img_path, model, transform):
    to_pil = T.ToPILImage()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        z, *_ = model.encode(x)
        recon = model.decode(z)
    
    recon = (recon.clamp(-1,1) + 1)/2  # [-1,1] -> [0,1]
    x_vis = (x.clamp(-1,1)+1)/2        # input normalizzato [0,1]

    # Calcola MSE tra input e output
    mse = F.mse_loss(x_vis, recon).item()

    # Visualizza input vs output
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].imshow(to_pil(x_vis[0].cpu()))
    axs[0].set_title("Input")
    axs[0].axis('off')

    axs[1].imshow(to_pil(recon[0].cpu()))
    axs[1].set_title(f"Output\nMSE={mse:.4f}")
    axs[1].axis('off')
    plt.show()

    return mse



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
        x1 = max(0, int(x - dx))
        y1 = max(0, int(y - dy))
        x2 = min(W, int(x + w + dx))
        y2 = min(H, int(y + h + dy))
        
        # Set the region to 1.0 (white)
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
            x = batch["image"].to(device)

            # Denormalize to [0, 1] for detection
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



def compute_explicit_codes(
    codes_forget, 
    codes_retain, 
    masks_forget=None,
    codebook_size=8192, 
    K=100,
    threshold=None
):
    """
    Compute explicit score for every code in codebook.
    Handles both simple frequency and spatial-aware (masked) frequency.
    
    Args:
        threshold (float, optional): If provided, returns all codes with a score 
                                           greater than this value, ignoring K.
    """
    device = codes_forget.device
    count_forget = torch.zeros(codebook_size, device=device)
    count_retain = torch.zeros(codebook_size, device=device)
    
    # Count Forget Set
    if masks_forget is not None:
        print("Computing scores using Spatial Masks...")
        flat_codes = codes_forget.reshape(-1)
        flat_masks = masks_forget.reshape(-1)
        
        relevant_codes = flat_codes[flat_masks > 0.5]
        unique, counts = relevant_codes.unique(return_counts=True)
        count_forget.index_add_(0, unique.long(), counts.float())
    else:
        print("Computing scores using Simple Frequency...")
        unique, counts = codes_forget.reshape(-1).unique(return_counts=True)
        count_forget.index_add_(0, unique.long(), counts.float())
        
    # Count Retain Set
    unique_r, counts_r = codes_retain.reshape(-1).unique(return_counts=True)
    count_retain.index_add_(0, unique_r.long(), counts_r.float())
    
    # Compute Score
    # Score = Frequency(Forget) / Frequency(Retain)
    scores = (count_forget + 1e-8) / (count_retain + 1e-8)
    
    # Select Codes
    if threshold is not None:
        # Filter by threshold
        explicit_codes = torch.where(scores > threshold)[0].tolist()
        print(f"Selected {len(explicit_codes)} codes above threshold {threshold}")
    else:
        # Filter by top K
        explicit_codes = torch.topk(scores, K).indices.tolist()
        
    return explicit_codes, scores



def get_safe_codes(codes_retain, explicit_codes, min_freq=10):
    """Find codes that are frequent in Retain but NOT in Explicit list."""
    flat = codes_retain.reshape(-1)
    unique, counts = flat.unique(return_counts=True)
    
    frequent = unique[counts >= min_freq].tolist()
    explicit_set = set(explicit_codes)
    
    safe = [c for c in frequent if c not in explicit_set]
    return safe



def load_image_to_tensor(path, image_size=256, device="cuda"):

    img = Image.open(path).convert("RGB")

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # batch dimension
    return img_tensor


def reconstruct_with_spatial_patching(
    model, 
    x, 
    detector,
    replacement_code, 
    device="cuda",
    expand=0.0,
    min_score=0.5
):
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        # ENCODE
        z_q, _, (_, _, indices) = model.encode(x)
        
        if indices.dim() == 2:
            indices = indices.squeeze(-1)
        
        if indices.dim() == 1:
            B = z_q.shape[0]
            H_latent = W_latent = 32
            indices = indices.view(B, H_latent, W_latent)
        
        B, H_latent, W_latent = indices.shape
        
        # CALCULATE MASK
        mask = torch.zeros_like(indices, dtype=torch.bool)
        
        # Convert tensor image to BGR numpy for NudeNet
        x_img = x[0].permute(1, 2, 0).cpu().numpy()
        x_img = ((x_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)
        
        detections = detector.detect(img_bgr)
        TARGET_CLASSES = ['FEMALE_BREAST_EXPOSED']
        
        # Calculate scaling factors (Image Pixels -> Latent Grid)
        scale_h = H_latent / x.shape[2]
        scale_w = W_latent / x.shape[3]

        found_explicit = False
        for d in detections:
            if d['score'] >= min_score and d['class'] in TARGET_CLASSES:
                found_explicit = True
                x_box, y_box, w_box, h_box = d['box']
                
                # Apply expansion
                dx, dy = int(w_box * expand), int(h_box * expand)
                x1 = max(0, x_box - dx)
                y1 = max(0, y_box - dy)
                x2 = min(x.shape[3], x_box + w_box + dx)
                y2 = min(x.shape[2], y_box + h_box + dy)

                # Map to latent coordinates
                lx1, ly1 = int(x1 * scale_w), int(y1 * scale_h)
                lx2, ly2 = int(x2 * scale_w), int(y2 * scale_h)
                
                # Ensure at least 1 index is covered
                lx2, ly2 = max(lx2, lx1 + 1), max(ly2, ly1 + 1)
                
                mask[0, ly1:ly2, lx1:lx2] = True

        num_substitutions = mask.sum().item()
        new_indices = indices.clone()
        
        # REPLACE
        if num_substitutions > 0:
            if isinstance(replacement_code, list) or isinstance(replacement_code, torch.Tensor):
                # Ensure it's a tensor
                if isinstance(replacement_code, list):
                    replacement_codes_tensor = torch.tensor(replacement_code, device=device, dtype=indices.dtype)
                else:
                    replacement_codes_tensor = replacement_code.to(device).type(indices.dtype)
                    
                mask_positions = mask.nonzero(as_tuple=True)
                random_replacements = replacement_codes_tensor[
                    torch.randint(0, len(replacement_codes_tensor), (num_substitutions,), device=device)
                ]
                new_indices[mask_positions] = random_replacements
            elif replacement_code is False:
                new_indices[mask] = torch.zeros_like(new_indices[mask])
            else:
                new_indices[mask] = replacement_code
        
        # DECODE
        new_indices_flat = new_indices.view(-1)
        z_q_modified = model.quantize.embed(new_indices_flat) 
        z_q_modified = z_q_modified.view(B, H_latent, W_latent, -1)
        z_q_modified = z_q_modified.permute(0, 3, 1, 2)

        x_recon_modified = model.decode(z_q_modified)

        return x_recon_modified, new_indices, mask, z_q, z_q_modified
    

def tensor_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    img = tensor.detach().cpu().float().numpy()
    
    img = np.transpose(img, (1, 2, 0))
    
    img = np.clip(img, -1, 1)
    img = (img + 1) / 2
    
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)


def reconstruct_from_codes(model, code_grid, device="cuda"):
    """
    Reconstructs an image from a grid of latent codes. 
    """
    model.eval()
    
    code_grid = code_grid.to(device)

    if code_grid.dim() == 2:
        code_grid = code_grid.unsqueeze(0)   

    B, H, W = code_grid.shape

    with torch.no_grad():
        flat = code_grid.reshape(-1)

        z_q = model.quantize.embed(flat) 

        emb_dim = z_q.shape[-1]
        z_q = z_q.view(B, H, W, emb_dim).permute(0, 3, 1, 2)

        x_recon = model.decode(z_q)

    return x_recon


def generate_nearest_neighbor_map(model, explicit_codes, safe_codes, device="cuda"):
    """
    Creates a dictionary mapping {explicit_code_idx -> nearest_safe_code_idx}
    based on embedding vector distance.
    """
    print(f"Mapping {len(explicit_codes)} explicit codes to {len(safe_codes)} safe candidates...")
    
    # Get Codebook Weights
    codebook = model.quantize.embed.weight.data.to(device)
    
    # Extract Vectors
    explicit_indices_t = torch.tensor(explicit_codes, device=device, dtype=torch.long)
    safe_indices_t = torch.tensor(safe_codes, device=device, dtype=torch.long)
    
    explicit_vectors = codebook[explicit_indices_t] # [K, Dim]
    safe_vectors = codebook[safe_indices_t]         # [M, Dim]
    
    # Compute Euclidean Distances (Pairwise)
    dists = torch.cdist(explicit_vectors, safe_vectors, p=2)
    
    # Find Nearest Neighbor for each Explicit Code
    min_vals, min_indices = torch.min(dists, dim=1)
    
    # Build the Lookup Tensor
    vocab_size = codebook.shape[0]
    lookup_tensor = torch.arange(vocab_size, device=device)
    
    # Overwrite the explicit entries
    nearest_safe_codes = safe_indices_t[min_indices]
    lookup_tensor[explicit_indices_t] = nearest_safe_codes
    
    return lookup_tensor

def reconstruct_with_code_substitution(
    model, 
    x, 
    negative_codes, 
    replacement_code, 
    device="cuda"
):
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        z_q, _, (_, _, indices) = model.encode(x)
        
        if indices.dim() == 2:
            indices = indices.squeeze(-1)
        
        if indices.dim() == 1:
            B = z_q.shape[0]
            H_latent = z_q.shape[2]
            W_latent = z_q.shape[3]
            indices = indices.view(B, H_latent, W_latent)
        
        B, H_latent, W_latent = indices.shape
        
        if not isinstance(negative_codes, list):
            negative_codes = [negative_codes]
        
        if len(negative_codes) == 0:
            x_recon_normal = model.decode(z_q)
            return x_recon_normal, indices, torch.zeros_like(indices, dtype=torch.bool)
        
        negative_codes_tensor = torch.tensor(negative_codes, device=device, dtype=indices.dtype)
        mask = (indices.unsqueeze(-1) == negative_codes_tensor.view(1, 1, 1, -1)).any(dim=-1)
        
        num_substitutions = mask.sum().item()
        new_indices = indices.clone()
        
        if num_substitutions > 0:
            if isinstance(replacement_code, list):
                replacement_codes_tensor = torch.tensor(replacement_code, device=device, dtype=indices.dtype)
                mask_positions = mask.nonzero(as_tuple=True)
                random_replacements = replacement_codes_tensor[
                    torch.randint(0, len(replacement_codes_tensor), (num_substitutions,), device=device)
                ]
                new_indices[mask_positions] = random_replacements
            elif replacement_code is False:
                new_indices[mask] = torch.zeros_like(new_indices[mask])
            else:
                new_indices[mask] = replacement_code
        
        new_indices_flat = new_indices.view(-1)
        z_q_modified = model.quantize.embed(new_indices_flat)  
        z_q_modified = z_q_modified.view(B, H_latent, W_latent, -1)
        z_q_modified = z_q_modified.permute(0, 3, 1, 2)

        x_recon_modified = model.decode(z_q_modified)

        return x_recon_modified, new_indices, mask


def reconstruct_with_learned_map(
    model, 
    x, 
    detector,          
    lookup_tensor,
    device="cuda",
    expand=0.0,
    min_score=0.5
):
    """
    Reconstructs an image while replacing explicit latent codes with their 
    nearest 'safe' neighbors as defined in 'lookup_tensor'.
    
    Args:
        model: The VQGAN/GumbelVQ model.
        x: Input tensor [1, 3, H, W] (normalized -1 to 1).
        detector: The NudeNet detector instance.
        lookup_tensor: A torch tensor where lookup_tensor[original_code] = replacement_code.
        device: 'cuda' or 'cpu'.
        expand: How much to pad the detected box (0.3 = 30%).
        min_score: Minimum confidence for detections.
    """
    model.eval()
    x = x.to(device)
    lookup_tensor = lookup_tensor.to(device)

    with torch.no_grad():
        z_q, _, (_, _, indices) = model.encode(x)
        
        # Handle index dimensions: Ensure [B, H, W]
        if indices.dim() == 2:
            indices = indices.squeeze(-1)
        
        if indices.dim() == 1:
            B = z_q.shape[0]
            H_latent = z_q.shape[2]
            W_latent = z_q.shape[3]
            indices = indices.view(B, H_latent, W_latent)
        
        B, H_latent, W_latent = indices.shape
        
        # Create a blank boolean mask
        mask = torch.zeros_like(indices, dtype=torch.bool)
        
        # Convert tensor image to BGR numpy for NudeNet
        x_img = x[0].permute(1, 2, 0).cpu().numpy()
        x_img = ((x_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)
        
        # Run Detection
        detections = detector.detect(img_bgr)
        TARGET_CLASSES = ['FEMALE_BREAST_EXPOSED']
        
        # Calculate scaling factors (Image Pixels -> Latent Grid)
        scale_h = H_latent / x.shape[2]
        scale_w = W_latent / x.shape[3]

        found_explicit = False
        for d in detections:
            if d['score'] >= min_score and d['class'] in TARGET_CLASSES:
                found_explicit = True
                x_box, y_box, w_box, h_box = d['box']
                
                # Apply expansion
                dx, dy = int(w_box * expand), int(h_box * expand)
                x1 = max(0, x_box - dx)
                y1 = max(0, y_box - dy)
                x2 = min(x.shape[3], x_box + w_box + dx)
                y2 = min(x.shape[2], y_box + h_box + dy)

                # Map to latent coordinates
                lx1, ly1 = int(x1 * scale_w), int(y1 * scale_h)
                lx2, ly2 = int(x2 * scale_w), int(y2 * scale_h)
                
                # Ensure at least 1 index is covered to avoid slicing errors
                lx2, ly2 = max(lx2, lx1 + 1), max(ly2, ly1 + 1)
                
                # Update the mask
                mask[0, ly1:ly2, lx1:lx2] = True

        final_indices = indices.clone()
        
        if found_explicit:
            # Translate the ENTIRE image using the safe map
            safe_version_indices = lookup_tensor[indices]
            
            # Selectively apply replacement only inside the mask
            # mask=True -> take from safe_version, mask=False -> keep original
            final_indices = torch.where(mask, safe_version_indices, indices)
            
            # Calculate the number of changed indices
            num_changed_indices = torch.sum(final_indices != indices).item()
            print(f"Number of latent codes changed: {num_changed_indices}")
        
        # Flatten for embedding lookup
        new_indices_flat = final_indices.view(-1)
        
        # Get embeddings
        z_q_modified = model.quantize.embed(new_indices_flat) 
        
        # Reshape embeddings to [B, C, H, W] for the decoder
        z_q_modified = z_q_modified.view(B, H_latent, W_latent, -1)
        z_q_modified = z_q_modified.permute(0, 3, 1, 2)

        # Generate final image
        x_recon_modified = model.decode(z_q_modified)

        return x_recon_modified, final_indices, mask, z_q, z_q_modified
    

def inject_and_decode(model, target_codes, source_codes, mask):
    """
    Mixes codes and immediately decodes them to an image.
    Robustly handles GumbelVQ / Taming Transformers dimensions.
    """
    # Inject the codes
    binary_mask = (mask > 0.5).to(target_codes.device)
    
    mixed_codes = target_codes.clone()
    mixed_codes[binary_mask] = source_codes[binary_mask]
    
    # Determine Embedding Dimension (C)    
    C = None
    
    # Direct attribute on quantizer (Standard)
    if hasattr(model.quantize, 'embed_dim'):
        C = model.quantize.embed_dim
        
    # Look at the internal Embedding layer
    elif hasattr(model.quantize, 'embed') and hasattr(model.quantize.embed, 'embedding_dim'):
        C = model.quantize.embed.embedding_dim
        
    # Fallback to input z_channels
    elif hasattr(model.quantize, 'z_channels'):
        C = model.quantize.z_channels
        
    # Last resort - inspect projection layer
    elif hasattr(model.quantize, 'proj'):
        if isinstance(model.quantize.proj, nn.Conv2d):
            C = model.quantize.proj.out_channels 
        else:
            C = model.quantize.proj.out_features
            
    if C is None:
        raise AttributeError("Could not determine embedding dimension (C). Please check model.quantize attributes.")

    # Decode
    B, H, W = mixed_codes.shape
    
    with torch.no_grad():
        # Flatten indices as expected by get_codebook_entry
        flat_indices = mixed_codes.reshape(-1)
        
        # Retrieve vectors: (B, H, W, C)
        z_q = model.quantize.get_codebook_entry(flat_indices, shape=(B, H, W, C))
        
        
        mixed_image = model.decode(z_q)
        
    return mixed_image

# ============================================================================
# Patches
# ============================================================================


def extract_patches(tensor, ps=3):
    if tensor.dim() == 3:
        B, H, W = tensor.shape
        # unfold H then W
        patches = tensor.unfold(1, ps, 1).unfold(2, ps, 1)
        # shape: [B, H_out, W_out, ps, ps]
        return patches.reshape(-1, ps * ps)
    return tensor


def score_patches(
    patches_forget,
    patches_retain,
    min_freq=5
):
    """
    Computes explicitness scores for unique latent patches.

    Args:
        patches_forget: Tensor [Nf, ps*ps]
        patches_retain: Tensor [Nr, ps*ps]
        min_freq: minimum frequency in forget set to keep a patch

    Returns:
        unique_patches: Tensor [U, ps*ps]
        scores: Tensor [U]
        count_forget: Tensor [U]
        count_retain: Tensor [U]
    """

    # Concatenate for joint indexing
    all_patches = torch.cat([patches_forget, patches_retain], dim=0)

    #Identify unique patches
    unique_patches, inverse = torch.unique(
        all_patches,
        dim=0,
        return_inverse=True
    )

    # Split inverse indices
    nf = patches_forget.shape[0]
    idx_forget = inverse[:nf]
    idx_retain = inverse[nf:]

    # Count frequencies
    num_uniques = unique_patches.shape[0]
    count_forget = torch.bincount(idx_forget, minlength=num_uniques).float()
    count_retain = torch.bincount(idx_retain, minlength=num_uniques).float()

    # Filter by minimum forget frequency
    valid_mask = count_forget >= min_freq

    # Explicitness score
    scores = (count_forget + 1e-8) / (count_retain + 1e-8)

    return (
        unique_patches[valid_mask],
        scores[valid_mask],
        count_forget[valid_mask],
        count_retain[valid_mask]
    )

def select_explicit_patches(
    unique_patches,
    scores,
    K=None,
    threshold=None
):
    """
    Selects explicit patches based on score ranking or threshold.

    Args:
        unique_patches: Tensor [U, ps*ps]
        scores: Tensor [U]
        K: optional top-K selection
        threshold: optional score threshold

    Returns:
        explicit_patches: List[tuple]
        explicit_scores: Dict {patch_id: score}
    """
    assert K is not None or threshold is not None, \
        "Either K or threshold must be provided"

    # Apply threshold if given
    if threshold is not None:
        mask = scores > threshold
        unique_patches = unique_patches[mask]
        scores = scores[mask]

    # Apply top-K if given
    if K is not None:
        sorted_scores, sort_idx = torch.sort(scores, descending=True)
        top_idx = sort_idx[:K]
        unique_patches = unique_patches[top_idx]
        scores = sorted_scores[:K]

    explicit_patches = [tuple(p.tolist()) for p in unique_patches]
    explicit_scores = {
        tuple(p.tolist()): s.item()
        for p, s in zip(unique_patches, scores)
    }

    return explicit_patches, explicit_scores


def extract_patch_ids(
    codes, 
    patch_size=3,
    return_locations=False
):
    """
    Extracts patch IDs from discrete latent grids.

    Args:
        codes: Tensor [B, H, W] of VQ indices
        patch_size: size of the square patch (default 3)
        return_locations: whether to also return (b, i, j) locations

    Returns:
        patch_ids: list of patch_id tuples
        locations (optional): list of (b, i, j) top-left positions
    """
    B, H, W = codes.shape
    ps = patch_size

    patch_ids = []
    locations = []

    for b in range(B):
        grid = codes[b]
        for i in range(H - ps + 1):
            for j in range(W - ps + 1):
                patch = grid[i:i+ps, j:j+ps]
                patch_id = tuple(patch.reshape(-1).tolist())
                patch_ids.append(patch_id)

                if return_locations:
                    locations.append((b, i, j))

    if return_locations:
        return patch_ids, locations
    return patch_ids


def compute_scores_from_counts(count_forget, count_retain):
    return (count_forget + 1e-8) / (count_retain + 1e-8)

def embed_patch(patch_id, codebook):
    """
    Converts a patch_id (tuple of indices) into a single embedding vector.

    Args:
        patch_id: tuple of length ps*ps
        codebook: Tensor [V, D]

    Returns:
        Tensor [D]
    """
    indices = torch.tensor(patch_id, device=codebook.device)
    vectors = codebook[indices]  # [9, D]
    return vectors.mean(dim=0)


def build_patch_replacement_map(
    model,
    explicit_patches,
    safe_patches,
    device="cuda"
):
    """
    Maps explicit patches to nearest safe patches in embedding space.

    Returns:
        dict {explicit_patch_id -> safe_patch_id}
    """
    codebook = model.quantize.embed.weight.data.to(device)

    explicit_vecs = torch.stack([
        embed_patch(p, codebook) for p in explicit_patches
    ])  # [Ne, D]

    safe_vecs = torch.stack([
        embed_patch(p, codebook) for p in safe_patches
    ])  # [Ns, D]

    dists = torch.cdist(explicit_vecs, safe_vecs, p=2)
    nn_indices = torch.argmin(dists, dim=1)

    replacement_map = {
        explicit_patches[i]: safe_patches[nn_indices[i].item()]
        for i in range(len(explicit_patches))
    }

    return replacement_map

def reconstruct_with_patch_unlearning(
    model,
    x,
    explicit_patch_set,
    patch_replacement_map,
    device="cuda",
    patch_size=3
):
    """
    Reconstructs an image by replacing explicit latent patches with
    their nearest safe neighbors (center-code replacement only).

    Args:
        model: pretrained VQGAN (taming-transformers)
        x: input tensor [1, 3, H, W] in [-1, 1]
        explicit_patch_set: set of explicit patch_id tuples
        patch_replacement_map: dict {explicit_patch_id -> safe_patch_id}
        device: cuda or cpu
        patch_size: spatial patch size (default 3)

    Returns:
        x_recon_modified: unlearned image
        final_indices: modified latent indices [1, H_lat, W_lat]
        patch_mask: boolean mask of replaced centers
    """
    model.eval()
    x = x.to(device)
    ps = patch_size
    center = ps // 2

    with torch.no_grad():
        # ENCODE
        z_q, _, (_, _, indices) = model.encode(x)

        if indices.dim() == 2:
            indices = indices.squeeze(-1)

        if indices.dim() == 1:
            B = z_q.shape[0]
            H_lat = z_q.shape[2]
            W_lat = z_q.shape[3]
            indices = indices.view(B, H_lat, W_lat)

        B, H_lat, W_lat = indices.shape
        assert B == 1, "This function assumes batch size = 1"

        # Clone indices for modification
        final_indices = indices.clone()

        # Track which centers are modified
        patch_mask = torch.zeros_like(indices, dtype=torch.bool)

        # SLIDE OVER PATCHES
        grid = indices[0]

        for i in range(H_lat - ps + 1):
            for j in range(W_lat - ps + 1):
                patch = grid[i:i+ps, j:j+ps]
                patch_id = tuple(patch.reshape(-1).tolist())

                if patch_id not in explicit_patch_set:
                    continue

                # Get nearest safe patch
                safe_patch_id = patch_replacement_map[patch_id]

                # Replace ONLY the center code
                safe_patch = torch.tensor(
                    safe_patch_id,
                    device=device,
                    dtype=indices.dtype
                ).view(ps, ps)

                ci = i + center
                cj = j + center

                final_indices[0, ci, cj] = safe_patch[center, center]
                patch_mask[0, ci, cj] = True

        num_changed = patch_mask.sum().item()
        print(f"Number of center-code replacements: {num_changed}")

        # DECODE
        flat_indices = final_indices.view(-1)
        z_q_modified = model.quantize.embed(flat_indices)
        z_q_modified = z_q_modified.view(B, H_lat, W_lat, -1)
        z_q_modified = z_q_modified.permute(0, 3, 1, 2)

        x_recon_modified = model.decode(z_q_modified)

        return x_recon_modified, final_indices, patch_mask, z_q, z_q_modified


def gather_batch(dataloaders, max_images=1000, device='cuda'):
    """
    Raccoglie sia i codes che le labels dai dataloader.
    Ritorna:
      - codes:  [N, 32, 32]
      - labels: [N]
    """
    all_codes = []
    all_labels = []
    count = 0

    for loader in dataloaders:
        for batch in loader:
            codes = batch["codes"]          # [B,32,32]
            labels = batch["label"]         # [B]

            all_codes.append(codes)
            all_labels.append(labels)

            count += codes.size(0)
            if count >= max_images:
                break

        if count >= max_images:
            break

    # concatenazione e taglio
    all_codes = torch.cat(all_codes, dim=0)[:max_images].to(device)
    all_labels = torch.cat(all_labels, dim=0)[:max_images].to(device)

    return all_codes, all_labels

def codebook_counterfactual_combinations(
    model,
    batch_codes,
    baseline_token=0,
    topk_single=50,
    max_comb=3,
    samples_per_comb=1000,
    device='cuda',
    batch_size=16
):
    """
    Identifies the most critical codebook entries and their combinations
    for counterfactual analysis using a classification model.

    """
    model.eval()
    batch_codes = batch_codes.to(device)
    B, H, W = batch_codes.shape

    with torch.no_grad():
        orig_probs = torch.sigmoid(model(batch_codes)).squeeze(-1)  # [B]

    unique_codes = torch.unique(batch_codes)
    importance_single = {}

    for code in tqdm(unique_codes, desc="Single code CF"):
        deltas = []
        for start in range(0, B, batch_size):
            modified = batch_codes[start:start+batch_size].clone()
            modified[modified == code] = baseline_token
            with torch.no_grad():
                new_probs = torch.sigmoid(model(modified)).squeeze(-1)
            deltas.append((orig_probs[start:start+batch_size] - new_probs).abs().cpu())
        importance_single[int(code.item())] = torch.cat(deltas).mean().item()

    top_codes = sorted(importance_single.items(), key=lambda x: x[1], reverse=True)
    top_codes = [c for c, _ in top_codes[:topk_single]]

    importance_combinations = {}

    for k in range(2, max_comb+1):
        print(f"\nSampling combinazioni di {k} codici...")
        for _ in tqdm(range(samples_per_comb)):
            comb = tuple(random.sample(top_codes, k))
            deltas = []
            for start in range(0, B, batch_size):
                modified = batch_codes[start:start+batch_size].clone()
                for code in comb:
                    modified[modified == code] = baseline_token
                with torch.no_grad():
                    new_probs = torch.sigmoid(model(modified)).squeeze(-1)
                deltas.append((orig_probs[start:start+batch_size] - new_probs).abs().cpu())
            importance_combinations[comb] = torch.cat(deltas).mean().item()

    top_combs = sorted(importance_combinations.items(), key=lambda x: x[1], reverse=True)

    return importance_single, top_codes, importance_combinations, top_combs


def counterfactual_grid_classifier(
    classifier,
    code_grids,            
    replacement_codes,      
    threshold=0.01,
    device="cuda"
):

    classifier.eval()

    if code_grids.dim() != 3:
        raise ValueError("code_grids deve essere [N,32,32].")

    code_grids = code_grids.to(device)
    N, H, W = code_grids.shape

    # Output
    new_grids = code_grids.clone()
    final_probs = torch.zeros(N, device=device)


    for idx in range(N):

        grid = new_grids[idx].unsqueeze(0)    # [1,32,32]

        with torch.no_grad():
            orig_prob = torch.sigmoid(classifier(grid))[0].item()

        print(f"\n=== Immagine {idx} ===")
        print(f"Prob iniziale classe 1 = {orig_prob:.4f}")

        if orig_prob >= 0.5:
            print("⚠ Già classificata come classe 1. Skip.")
            final_probs[idx] = orig_prob
            continue

        current_prob = orig_prob

        for rep_code in replacement_codes:

            best_increase = 0.0
            best_pos = None

            for i in range(H):
                for j in range(W):

                    old_val = grid[0, i, j].item()
                    if old_val == rep_code:
                        continue

                    tmp = grid.clone()
                    tmp[0, i, j] = rep_code

                    with torch.no_grad():
                        prob = torch.sigmoid(classifier(tmp))[0].item()

                    increase = prob - current_prob

                    if increase > best_increase:
                        best_increase = increase
                        best_pos = (i, j)

            if best_pos is None or best_increase < threshold:
                print(f"→ Nessun miglioramento con codice {rep_code}. STALLO.")
                break

            i, j = best_pos
            print(f"✓ Applying code {rep_code} at ({i},{j}) | +{best_increase:.4f}")

            grid[0, i, j] = rep_code

            with torch.no_grad():
                current_prob = torch.sigmoid(classifier(grid))[0].item()

            print(f" → Prob aggiornata = {current_prob:.4f}")

            if current_prob >= 0.5:
                print("✓ Classificata come classe 1!")
                break

        new_grids[idx] = grid[0]
        final_probs[idx] = current_prob

    return new_grids, final_probs


class VQGANCodeAnalyzer:
    """
    Analyzer for VQGAN codes extracted from masked regions.
    """

    def __init__(self, all_codes_linear, all_codes_masked_grid=None, codebook_size=1024):
        """
        Initialize analyzer with linear and optional spatial codes.
        """
        self.codes_per_mask = all_codes_linear
        self.codes_grid = all_codes_masked_grid
        self.all_codes = np.concatenate(all_codes_linear)
        self.codebook_size = codebook_size
        self.n_masks = len(all_codes_linear)

        print(f"Inizializzato analyzer con {len(self.all_codes)} codici totali")
        print(f"da {self.n_masks} maschere diverse")

        if all_codes_masked_grid is not None:
            print("Dati spaziali disponibili per analisi co-occorrenze\n")
        else:
            print("ATTENZIONE: Dati spaziali non disponibili, co-occorrenze semplificate\n")

    def analyze_frequency(self, top_k=50):
        """
        Analyze single-code frequency statistics.
        """
        print("=" * 70)
        print("ANALISI FREQUENZA CODICI")
        print("=" * 70)

        unique, counts = np.unique(self.all_codes, return_counts=True)
        freq_dict = dict(zip(unique, counts))
        sorted_codes = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

        total = len(self.all_codes)

        print("\nStatistiche Generali:")
        print(f" Codici totali: {total}")
        print(f" Codici unici: {len(unique)}")
        print(f" Copertura codebook: {len(unique)/self.codebook_size:.2%}")
        print(f" Codici inutilizzati: {self.codebook_size - len(unique)}")

        print(f"\nTop {min(top_k, len(sorted_codes))} Codici più Frequenti:")
        print(f"{'Rank':<6} {'Codice':<10} {'Freq':<10} {'%':<10} {'Cum %':<10}")
        print("-" * 60)

        cumulative = 0
        top_codes = []

        for i, (code, count) in enumerate(sorted_codes[:top_k], 1):
            pct = count / total * 100
            cumulative += pct
            top_codes.append(code)
            print(f"{i:<6} {code:<10} {count:<10} {pct:>6.2f}% {cumulative:>6.2f}%")

        top10_coverage = sum(c for _, c in sorted_codes[:10]) / total
        top50_coverage = sum(c for _, c in sorted_codes[:50]) / total

        print("\nConcentrazione (Pareto):")
        print(f" Top 10 codici coprono: {top10_coverage:.2%}")
        print(f" Top 50 codici coprono: {top50_coverage:.2%}")

        probs = counts / counts.sum()
        H = entropy(probs, base=2)
        max_entropy = np.log2(self.codebook_size)

        print("\nEntropia:")
        print(f" Entropia attuale: {H:.2f} bits")
        print(f" Entropia massima: {max_entropy:.2f} bits")
        print(f" Utilizzo codebook: {H/max_entropy:.2%}")

        return {
            'top_codes': top_codes,
            'frequencies': freq_dict,
            'sorted_codes': sorted_codes,
            'stats': {
                'total': total,
                'unique': len(unique),
                'top10_coverage': top10_coverage,
                'top50_coverage': top50_coverage,
                'entropy': H,
                'max_entropy': max_entropy
            }
        }

    def analyze_cooccurrence(self, top_k=30, use_spatial=True):
        """
        Analyze pair co-occurrence patterns.
        """
        print("\n" + "=" * 70)
        print("ANALISI CO-OCCORRENZE")
        print("=" * 70)

        pair_counter = Counter()

        if use_spatial and self.codes_grid is not None:
            print("Usando co-occorrenze SPAZIALI\n")

            for image_masks in self.codes_grid:
                for mask_2d in image_masks:
                    valid_coords = np.argwhere(mask_2d != -1)
                    if len(valid_coords) < 2:
                        continue

                    coord_to_code = {
                        tuple(c): mask_2d[c[0], c[1]] for c in valid_coords
                    }

                    for y, x in valid_coords:
                        center = mask_2d[y, x]
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y + dy, x + dx
                            if (ny, nx) in coord_to_code:
                                pair = tuple(sorted([center, coord_to_code[(ny, nx)]]))
                                pair_counter[pair] += 1
        else:
            print("Usando co-occorrenze LINEARI\n")
            for mask_codes in self.codes_per_mask:
                for i in range(len(mask_codes) - 1):
                    pair = tuple(sorted([mask_codes[i], mask_codes[i+1]]))
                    pair_counter[pair] += 1

        print(f"\nTop {top_k} Coppie:")
        print(f"{'Rank':<6} {'Coppia':<20} {'Freq':<10}")
        print("-" * 60)

        top_pairs = pair_counter.most_common(top_k)
        for i, (pair, count) in enumerate(top_pairs, 1):
            print(f"{i:<6} {str(pair):<20} {count:<10}")

        return {
            'pair_counter': pair_counter,
            'top_pairs': top_pairs,
            'spatial': use_spatial and self.codes_grid is not None
        }

    def analyze_mask_similarity(self, sample_size=None, use_weighted=True):
        """
        Analyze similarity across masks.
        """
        print("\n" + "=" * 70)
        print("ANALISI SIMILARITÀ MASCHERE")
        print("=" * 70)

        if sample_size and sample_size < len(self.codes_per_mask):
            idx = np.random.choice(len(self.codes_per_mask), sample_size, replace=False)
            masks = [self.codes_per_mask[i] for i in idx]
        else:
            masks = self.codes_per_mask

        n = len(masks)
        sim_matrix = np.zeros((n, n))

        if use_weighted:
            print("Cosine similarity su istogrammi\n")
            hists = []

            for codes in masks:
                hist = np.zeros(self.codebook_size)
                u, c = np.unique(codes, return_counts=True)
                hist[u] = c
                hist /= hist.sum()
                hists.append(hist)

            for i in range(n):
                for j in range(i, n):
                    sim = 1 - cosine(hists[i], hists[j])
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        else:
            print("Jaccard similarity\n")
            for i in range(n):
                si = set(masks[i])
                for j in range(i, n):
                    sj = set(masks[j])
                    sim = len(si & sj) / len(si | sj)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        avg_sim = np.mean(sim_matrix[np.triu_indices(n, 1)])
        print(f"Similarità media: {avg_sim:.3f}")

        return {
            'similarity_matrix': sim_matrix,
            'avg_similarity': avg_sim,
            'method': 'cosine' if use_weighted else 'jaccard'
        }

    def run_full_analysis(
        self,
        background_codes=None,
        plot=True,
        use_spatial=True,
        use_weighted_similarity=True
    ):
        """
        Run full analysis pipeline.
        """
        print("\n" + "=" * 70)
        print("ANALISI COMPLETA VQGAN")
        print("=" * 70)

        results = {}
        results['frequency'] = self.analyze_frequency()
        results['cooccurrence'] = self.analyze_cooccurrence(use_spatial=use_spatial)
        results['similarity'] = self.analyze_mask_similarity(
            use_weighted=use_weighted_similarity
        )

        print("\nANALISI COMPLETATA\n")
        return results


def analyze_saved_codes(codes_path, codes_grid_path=None, codebook_size=1024):
    """
    Load saved codes and run full analysis.
    """
    print("Caricamento codici...")
    all_codes_linear = np.load(codes_path, allow_pickle=True)

    all_codes_grid = None
    if codes_grid_path:
        all_codes_grid = np.load(codes_grid_path, allow_pickle=True)

    analyzer = VQGANCodeAnalyzer(
        all_codes_linear,
        all_codes_grid,
        codebook_size
    )

    results = analyzer.run_full_analysis()
    return analyzer, results


def extract_codes_from_whole_images(
    model,
    img_folder,
    device='cuda',
    max_images=None,
    return_per_image=False
):
    """
    Extract VQGAN codes from full images.
    """
    model.eval()
    model.to(device)

    img_files = [
        f for f in os.listdir(img_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    if max_images is not None:
        img_files = img_files[:max_images]

    print(f"Estrazione background da {len(img_files)} immagini")

    all_codes = []
    codes_per_image = []

    for fname in tqdm(img_files, desc="Background VQGAN"):
        img_path = os.path.join(img_folder, fname)

        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img) / 255.0

            img_tensor = torch.tensor(
                img_np,
                dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            img_tensor = 2.0 * img_tensor - 1.0

            with torch.no_grad():
                z, _, info = model.encode(img_tensor)
                code_indices = info[2][0].cpu().numpy().reshape(-1)

            all_codes.append(code_indices)
            codes_per_image.append(code_indices)

        except Exception as e:
            print(f"⚠️ Errore su {fname}: {e}")
            continue

    if not all_codes:
        raise RuntimeError("Nessun codice estratto per il background!")

    background_codes = np.concatenate(all_codes)

    print(f"✅ Background estratto:")
    print(f"   • Codici totali: {len(background_codes)}")
    print(f"   • Immagini processate: {len(all_codes)}")

    if return_per_image:
        return background_codes, codes_per_image

    return background_codes

def batch_extract_masked_codes(
    model,
    img_folder,
    device='cuda',
    max_images=None
):
    """
    Extract VQGAN codes under detected female breast regions.
    """

    detector = nudenet.NudeDetector()

    all_codes_linear = []
    all_codes_masked_grid = []
    metadata = []

    img_files = [
        f for f in os.listdir(img_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    if max_images is not None:
        img_files = img_files[:max_images]

    print(f"Processando {len(img_files)} immagini...")

    for fname in tqdm(img_files, desc="Estrazione codici"):
        img_path = os.path.join(img_folder, fname)

        try:
            img_pil = Image.open(img_path).convert('RGB')
            img_cv = cv2.imread(img_path)
            img_h, img_w = img_cv.shape[:2]

            results = detector.detect(img_path)
            breast_masks = []

            for r in results:
                if r['class'] == 'FEMALE_BREAST_EXPOSED':
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)

                    x, y, w, h = [int(v) for v in r['box']]
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)

                    mask[y:y+h, x:x+w] = 1
                    breast_masks.append(mask)

            if not breast_masks:
                continue

            img_tensor = torch.tensor(
                np.array(img_pil) / 255.0,
                dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            img_tensor = 2.0 * img_tensor - 1.0

            model.eval()
            with torch.no_grad():
                z, _, info = model.encode(img_tensor)

            code_indices = info[2][0]

            _, _, Hc, Wc = z.shape
            code_grid = code_indices.reshape(Hc, Wc).cpu().numpy()

            codes_for_image = []

            for mask_id, mask in enumerate(breast_masks):
                mask_resized = cv2.resize(
                    mask,
                    (Wc, Hc),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                masked_codes_2d = np.where(mask_resized, code_grid, -1)
                masked_codes_linear = code_grid[mask_resized]

                all_codes_linear.append(masked_codes_linear)
                codes_for_image.append(masked_codes_2d)

                metadata.append({
                    'filename': fname,
                    'mask_id': mask_id,
                    'grid_shape': (Hc, Wc),
                    'n_codes': len(masked_codes_linear),
                    'mask_coverage': mask_resized.sum() / (Hc * Wc)
                })

            all_codes_masked_grid.append(codes_for_image)

        except Exception as e:
            print(f"\nErrore processando {fname}: {e}")
            continue

    print(f"\nTotale maschere processate: {len(all_codes_linear)}")
    print(f"Totale immagini con maschere: {len(all_codes_masked_grid)}")

    return all_codes_linear, all_codes_masked_grid, metadata


def extract_local_pairs_vectorized(grid, neighborhood=1):
    """
    Extract local code pairs using vectorized operations.
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)
    
    B, H, W = grid.shape
    all_pairs = []
    
    offsets = []
    for dx in range(-neighborhood, neighborhood + 1):
        for dy in range(-neighborhood, neighborhood + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    
    for dx, dy in offsets:
        h_start = max(0, dx)
        h_end = min(H, H + dx)
        w_start = max(0, dy)
        w_end = min(W, W + dy)
        
        center = grid[:, h_start:h_end, w_start:w_end]
        neighbor = grid[:, h_start-dx:h_end-dx, w_start-dy:w_end-dy]
        
        pairs = torch.stack(
            [center.flatten(), neighbor.flatten()],
            dim=1
        )
        all_pairs.append(pairs)
    
    return torch.cat(all_pairs, dim=0)


def build_graph_from_images(
    tensor_list,
    K=8192,
    neighborhood=1,
    batch_size=32,
    chunk_size=100
):
    """
    Build a sparse co-occurrence graph from code grids.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_indices = []
    graph_values = []
    chunk_counter = 0

    total_images = sum(t.shape[0] for t in tensor_list)
    pbar = tqdm(
        total=total_images,
        desc="Processing images",
        dynamic_ncols=True
    )

    for batch_tensor in tensor_list:
        batch_tensor = batch_tensor.to(device, dtype=torch.int64)

        for i in range(0, batch_tensor.shape[0], batch_size):
            mini = batch_tensor[i:i + batch_size]

            pairs = extract_local_pairs_vectorized(
                mini,
                neighborhood
            )

            pairs = pairs.cpu()

            graph_indices.append(pairs.t().contiguous())
            graph_values.append(
                torch.ones(pairs.shape[0], dtype=torch.int32)
            )

            chunk_counter += 1
            pbar.update(mini.shape[0])

            if chunk_counter >= chunk_size:
                indices = torch.cat(graph_indices, dim=1)
                values = torch.cat(graph_values, dim=0)

                temp_graph = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=(K, K)
                ).coalesce()

                graph_indices = [temp_graph.indices()]
                graph_values = [temp_graph.values()]
                chunk_counter = 0

    pbar.close()

    indices = torch.cat(graph_indices, dim=1)
    values = torch.cat(graph_values, dim=0)

    graph = torch.sparse_coo_tensor(
        indices,
        values,
        size=(K, K)
    )

    return graph.coalesce()
