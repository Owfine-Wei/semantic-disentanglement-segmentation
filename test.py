import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

import fcn_model
import helpers.config as config
from data_sds_cityscapes import load_data
from helpers.visualize_val import decode_segmap, overlap_images, preprocess_image
from train import compute_consistency_loss

# ================= Configuration =================
MODEL_PATH = '/root/autodl-tmp/models/_1_9_2026_BLFPN2_.pth' 

# Hyperparameters requested
MODE = 'csg+origin'
SPLIT = 'train'
CSG_MODE = 'foreground'
BATCH_SIZE = 8 # Small batch for testing

# Loss parameters
ALPHA = 1.0
BETA = 0.0

# Visualization
OUTPUT_DIR = '/root/autodl-tmp/test_results/'

# =================================================

def denormalize(tensor, mean, std):
    """
    Inverse of preprocess_image normalization for visualization.
    tensor: (C, H, W)
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = fcn_model.get_model(num_classes=19, checkpoint=MODEL_PATH) # Assuming 19 classes for Cityscapes
    
    if os.path.exists(MODEL_PATH):
        # Handle 'module.' prefix if trained with DDP
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model = model.to(device)
    model.eval() # Use eval mode for testing (affects BN/Dropout)

    # 2. Load Dataloader
    print("Loading dataloader...")
    # Note: data_sds_cityscapes.load_data likely returns a DataLoader
    train_loader = load_data(
        root=config.DATA_DIR, 
        mode=MODE, 
        split=SPLIT, 
        csg_mode=CSG_MODE, 
        batch_size=BATCH_SIZE,
        distributed=False
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    print("Starting inference on one batch...")
    
    # 3. Get one batch
    # dataloader returns: images, labels, mask, origin_images, origin_labels
    data_iter = iter(train_loader)
    try:
        batch = next(data_iter)
    except StopIteration:
        print("Dataloader is empty.")
        return

    images, labels, mask, origin_images, origin_labels = batch

    # Move to device
    images = images.to(device)
    labels = labels.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    origin_images = origin_images.to(device)
    origin_labels = origin_labels.to(device, dtype=torch.long)

    # 4. Execute Logic
    # Note: optimizer.zero_grad() is removed as we are not training here.
    
    with torch.no_grad(): # Disable gradient calculation for testing
        # Forward pass
        outputs_img, _ = model(images)
        if MODE=='csg+origin':
            outputs_origin, _ = model(origin_images)
        
        # Calculate loss
        loss_img = criterion(outputs_img, labels.squeeze(1))
        
        integrated_loss = 0.0
        
        if MODE=='csg+origin':
            loss_origin = criterion(outputs_origin, origin_labels.squeeze(1))

            mask = mask.unsqueeze(1)

            outputs_origin_frozen = outputs_origin.detach()

            diff = (outputs_img - outputs_origin_frozen) * (1.0 - mask)

            # valid_pixels = torch.sum(1.0 - mask) * outputs_img.size(1)

            # consist_loss = torch.sum(diff**2) / (valid_pixels + 1e-6)

            consist_loss = compute_consistency_loss(outputs_img, outputs_origin_frozen, mask)

            integrated_loss = loss_img + ALPHA * consist_loss + BETA * loss_origin
            
            print("\n========== Loss Calculation Results ==========")
            print(f"Loss Img (CSG): {loss_img.item():.4f}")
            print(f"Loss Origin   : {loss_origin.item():.4f}")
            print(f"Consist Loss  : {consist_loss.item():.4f}")
            print(f"Integrated Loss: {integrated_loss.item():.4f}")
            print("==============================================\n")
        else:
            integrated_loss = loss_img
            print(f"Integrated Loss: {integrated_loss.item():.4f}")

        # 5. Visualization
        # We will visualize the first image in the batch
        print("Saving visualization results...")
        
        # Helper to get prediction map
        def get_pred_map(logits):
            pred = torch.argmax(logits, dim=1).cpu().numpy() # (B, H, W)
            return pred

        pred_img = get_pred_map(outputs_img)
        pred_origin = get_pred_map(outputs_origin)

        # Get visualizable images (denormalize)
        # Handle Normalization Stats correctly
        
        # Stats for Origin images
        mean_origin = np.array(config.RGB_MEAN['origin'][SPLIT])
        std_origin  = np.array(config.RGB_STD['origin'][SPLIT])

        # Stats for CSG images
        # User request: Use Origin stats for ALL denormalization
        mean_csg = mean_origin
        std_csg  = std_origin

        # We need to loop through the batch to save images
        for i in range(BATCH_SIZE): # Process max 4 images
            
            # --- Process CSG Image ---
            img_tensor = images[i].cpu()
            # Denormalize CSG
            # Now CSG Loader returns RGB.
            # So after denormalize, we get RGB directly.
            vis_img = denormalize(img_tensor, mean_csg, std_csg).permute(1, 2, 0).numpy()
            vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)
            
            # Visualization directly uses vis_img (RGB)
            vis_img_rgb = vis_img
            
            # Prediction Color
            seg_color_img = decode_segmap(pred_img[i], nc=256)

            # --- Mask Processing (Erase masked region in prediction) ---
            # mask is (B, 1, H, W), mask[i] is (1, H, W)
            # mask=1 means erased/invalid region
            mask_boolean = mask[i].squeeze(0).cpu().numpy().astype(bool) # (H, W)
            seg_color_img[mask_boolean] = [255, 255, 255] # Set to White

            overlap_img = overlap_images(vis_img_rgb, seg_color_img)
            
            # --- Process Origin Image ---
            origin_tensor = origin_images[i].cpu()
            # Denormalize Origin (This is RGB)
            vis_origin_rgb = denormalize(origin_tensor, mean_origin, std_origin).permute(1, 2, 0).numpy()
            vis_origin_rgb = np.clip(vis_origin_rgb * 255, 0, 255).astype(np.uint8)
            
            # For saving to disk with OpenCV (needs BGR)
            vis_origin_bgr = cv2.cvtColor(vis_origin_rgb, cv2.COLOR_RGB2BGR)

            # Prediction Color
            seg_color_origin = decode_segmap(pred_origin[i], nc=256) # Returns RGB
            overlap_origin = overlap_images(vis_origin_rgb, seg_color_origin)
            
            # --- Mask Visualization ---
            mask_vis = mask[i].squeeze(0).cpu().numpy() # (H,W)
            mask_vis = (mask_vis * 255).astype(np.uint8)
            
            # --- Diff Visualization ---
            diff_map = diff[i].cpu().numpy() # (C, H, W)
            diff_map = np.sum(np.abs(diff_map), axis=0) # (H, W)
            diff_map = (diff_map / (diff_map.max() + 1e-8) * 255).astype(np.uint8)
            diff_heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

            # --- Ground Truth Visualization ---
            label_csg = labels[i].squeeze().cpu().numpy()
            label_origin = origin_labels[i].squeeze().cpu().numpy()

            gt_color_csg = decode_segmap(label_csg, nc=256)
            gt_color_origin = decode_segmap(label_origin, nc=256)

            gt_overlap_csg = overlap_images(vis_img_rgb, gt_color_csg)
            gt_overlap_origin = overlap_images(vis_origin_rgb, gt_color_origin)

            # --- Plotting ---
            filename = f"batch_{i}"
            
            # Updated to 2 rows, 5 columns
            fig, axes = plt.subplots(2, 5, figsize=(30, 10))
            
            # Row 1: CSG Image
            axes[0,0].imshow(vis_img_rgb) # Now RGB
            axes[0,0].set_title("Input (CSG)")
            
            axes[0,1].imshow(seg_color_img)
            axes[0,1].set_title("Prediction (CSG) - Masked")
            
            axes[0,2].imshow(overlap_img)
            axes[0,2].set_title("Overlap (CSG)")

            axes[0,3].imshow(gt_color_csg)
            axes[0,3].set_title("GT (CSG)")

            axes[0,4].imshow(gt_overlap_csg)
            axes[0,4].set_title("GT Overlap (CSG)")

            # Row 2: Origin Image
            axes[1,0].imshow(vis_origin_rgb)
            axes[1,0].set_title("Input (Origin)")
            
            axes[1,1].imshow(seg_color_origin)
            axes[1,1].set_title("Prediction (Origin)")
            
            axes[1,2].imshow(overlap_origin)
            axes[1,2].set_title("Overlap (Origin)")

            axes[1,3].imshow(gt_color_origin)
            axes[1,3].set_title("GT (Origin)")

            axes[1,4].imshow(gt_overlap_origin)
            axes[1,4].set_title("GT Overlap (Origin)")
            
            # Save
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"{filename}_comparison.png")
            plt.savefig(save_path)
            plt.close()

            # Save Mask and Diff separately
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_mask.png"), mask_vis)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_diff.png"), diff_heatmap)
            
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
