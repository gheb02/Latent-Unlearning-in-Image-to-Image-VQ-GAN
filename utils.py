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
from collections import Counter



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
