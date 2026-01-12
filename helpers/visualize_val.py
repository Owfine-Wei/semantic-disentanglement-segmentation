import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from fcn_model import get_model
import helpers.config as config

# ======== Modified by User ========
model_path = ''
# ==================================

# To_Tensor and Normalize the Image 
def preprocess_image(image_path, mean, std):
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy() 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0), original_img 


# Convert numpredictions(numbers) into predictions(colors)
def decode_segmap(image, nc=21):
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l

        if l in config.CLASS_COLORS:
            r[idx] = config.CLASS_COLORS[l][0]
            g[idx] = config.CLASS_COLORS[l][1]
            b[idx] = config.CLASS_COLORS[l][2]

        else:
            r[idx] = 0
            g[idx] = 0
            b[idx] = 0
            
    rgb = np.stack([r, g, b], axis=2)
    return rgb


# Overlap the original image with predictions(colors)
def overlap_images(original_img, seg_map, alpha=0.5):
    
    original_np = np.array(original_img)
    
    if original_np.shape[:2] != seg_map.shape[:2]:
        seg_map = cv2.resize(seg_map, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = (original_np * (1 - alpha) + seg_map * alpha).astype(np.uint8)
    return overlay

# 3. Main

def visualize_val(model, device, img_root, output_dir, mean, std):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Collect Val Images
    test_images = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print(f"No images found in {img_root}")
        return

    # Choose Images Randomly
    num_samples = 20
    selected_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    print(f"Processing {len(selected_images)} images...")

    with torch.no_grad():
        for i, img_path in enumerate(selected_images):
            print(f"Processing: {img_path}")
            
            img_tensor, original_img = preprocess_image(img_path,mean,std)
            img_tensor = img_tensor.to(device)
            
            output, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            seg_color = decode_segmap(pred, nc=256) 
            
            overlap = overlap_images(original_img, seg_color)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(save_path, cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(seg_color)
            axes[1].set_title("Segmentation")
            axes[1].axis('off')
            
            axes[2].imshow(overlap)
            axes[2].set_title("Overlap")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{filename}"))
            plt.close()
            
    print(f"Done! Results saved to {output_dir}")

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")

    model = get_model(num_classes=19, checkpoint=model_path) 

    model.to(device)
    model.eval()

    # Test
    visualize_val(model, device, config.VAL_IMGS_DIR, config.VISUAL_IMGS_DIR, config.RGB_MEAN, config.RGB_STD)


if __name__ == '__main__':
    main()
