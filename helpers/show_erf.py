"""
Utilities to visualize input-gradient contributions (ERF) for segmentation.

This module loads a trained model, computes input gradients for a
selected semantic class mask, normalizes the contribution map, and writes
several visualization images (heatmap, highlight mask, overlays).
"""

import torch
import torch.nn.functional as F
import helpers.config as config
import fcn_model
import cv2
import numpy as np
import os


# ======== Modified by User ========

IMG_DIR = "/root/autodl-tmp/data/Cityscapes/leftImg8bit/val"
LABEL_DIR = "/root/autodl-tmp/data/Cityscapes/gtFine/val"

semantic = 'car'

out_dir = "/root/autodl-tmp/outputs/val_erf"

checkpoint_path = "/root/autodl-tmp/models/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth"

# ==================================

def get_imgs(img_dir, label_dir):
	"""
	Walk `img_dir` and `label_dir` recursively and return two lists:
	- imgs: sorted list of full image file paths found under `img_dir`
	- labels: sorted list of full label file paths found under `label_dir`

	This uses `os.walk` to traverse city subdirectories under the provided
	`.../val/` roots. It supports common image extensions and returns
	consistently sorted lists so downstream code can rely on ordering.
	"""

	# helper to collect files with allowed extensions
	def _collect_files(root_dir):
		collected = []
		for droot, _, files in os.walk(root_dir):
			for fn in files:
				if fn.lower().endswith(".png"):
					collected.append(os.path.join(droot, fn))
		collected.sort()
		return collected

	imgs = _collect_files(img_dir)
	labels = _collect_files(label_dir)

	return imgs, labels


def img_process(img_path):
	"""
	Read image, normalize by config mean/std and return CHW tensor.

	Returns a float32 tensor shaped (1, C, H, W).
	"""

	img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Image not found: {img_path}")

	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

	mean = np.array(config.RGB_MEAN, dtype=np.float32).reshape(1, 1, 3)
	std = np.array(config.RGB_STD, dtype=np.float32).reshape(1, 1, 3)
	img_norm = (img_rgb - mean) / std

	img_chw = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)

	img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

	return img_tensor


def get_grad_mask(label_path, semantic):
	"""
	Load label image and return a binary mask tensor for `semantic`.

	The mask has shape (1,1,H,W) and dtype long, marking pixels equal to
	the trainId of the requested semantic class.
	"""

	label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
	if label is None:
		raise FileNotFoundError(f"Label not found: {label_path}")

	if semantic not in config.TRAIN_ID_DICT:
		raise ValueError(f"Semantic '{semantic}' not found in config.TRAIN_ID_DICT")
	trainid = config.TRAIN_ID_DICT[semantic]

	mask_np = (label == trainid).astype(np.uint8)

	mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(torch.long)

	return mask_tensor


def get_contirb(checkpoint_path, input_tensor, grad_mask=None):
	"""
	Compute normalized input-gradient contribution map for `grad_mask`.

	If `grad_mask` is None the function returns raw logits. When provided,
	the mask is expanded to match logits channels and used as `grad_outputs`
	in `torch.autograd.grad` to compute per-pixel input gradients.
	Returns a normalized contribution map in [0,1] with shape (1,1,H,W).
	"""

	if not isinstance(input_tensor, torch.Tensor):
		raise TypeError('input_tensor must be a torch.Tensor')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device_str = str(device)

	model = fcn_model.get_model(checkpoint=checkpoint_path, device=device_str)
	model = model.to(device)
	model.eval()

	# Prepare input with grad enabled
	input_var = input_tensor.to(device).requires_grad_(True)

	out = model(input_var)

	# handle models returning tuple/list or dict
	if isinstance(out, (tuple, list)):
		logits = out[0]
	elif isinstance(out, dict) and 'logits' in out:
		logits = out['logits']
	else:
		logits = out

	if logits.dim() == 3:
		logits = logits.unsqueeze(0)

	_, _, h, w = logits.shape
	if (h, w) != config.IMG_SIZE:
		logits = F.interpolate(logits, size=config.IMG_SIZE, mode='bilinear', align_corners=False)

	# if no grad mask requested, return logits directly
	if grad_mask is None:
		return logits

	# normalize and reshape grad_mask to (B,1,H,W) or (B,C,H,W)
	if not torch.is_tensor(grad_mask):
		grad_mask = torch.from_numpy(np.array(grad_mask))
	grad_mask = grad_mask.to(device=logits.device)
	if grad_mask.dim() == 3:
		grad_mask = grad_mask.unsqueeze(1)
	if grad_mask.dim() != 4:
		raise ValueError('grad_mask must be shape (B,1,H,W) or (B,H,W)')

	# expand mask to logits channel dimension when needed
	if grad_mask.size(1) == 1 and logits.size(1) != 1:
		grad_for_logits = grad_mask.expand(-1, logits.size(1), -1, -1).to(dtype=logits.dtype)
	else:
		grad_for_logits = grad_mask.to(dtype=logits.dtype)

	# compute input gradients w.r.t. logits using autograd.grad
	input_grad = torch.autograd.grad(outputs=logits, inputs=input_var, grad_outputs=grad_for_logits, retain_graph=True, allow_unused=True)[0]

	if input_grad is None:
		raise RuntimeError('autograd.grad returned None; ensure logits depend on input and requires_grad is True on input.')

	contrib = input_grad.abs().sum(dim=1, keepdim=True)

	contrib = torch.log1p(contrib)
	contrib = contrib.to(torch.float32)
	B, C, H, W = contrib.shape
	tmp = contrib.view(B, -1)

	q_max = torch.quantile(tmp, 0.995, dim=1, keepdim=True).view(B, 1, 1, 1)
	q_min = contrib.amin(dim=[1, 2, 3], keepdim=True)

	eps = 1e-8
	contrib = torch.clamp(contrib, min=q_min, max=q_max)
	contrib = (contrib - q_min) / (q_max - q_min + eps)

	return contrib


def show_img(img_path, contrib, out_dir):
	"""
	Save visualization images: original, heatmap, highlight mask and overlays.

	`contrib` may be a tensor or numpy array; it will be resized/clipped to
	the original image size and used to create multiple outputs.
	"""

	# get group dir first
	file_prefix = (os.path.basename(img_path)).replace("_leftImg8bit.png","")
	group_dir = os.path.join(out_dir, file_prefix)
	os.makedirs(group_dir, exist_ok=True)

	# 0. read original image (RGB) and save
	img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Image not found: {img_path}")
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	H, W = img_rgb.shape[:2]

	# prepare contrib as HxW numpy array in 0-1
	if torch.is_tensor(contrib):
		c = contrib.detach().cpu().numpy()
	else:
		c = np.array(contrib)

	# support shapes: (1,1,H,W), (1,H,W), (H,W)
	if c.ndim == 4:
		c = c[0, 0]
	elif c.ndim == 3:
		# could be (1,H,W)
		if c.shape[0] == 1:
			c = c[0]
		else:
			# assume (H,W,C) unlikely, collapse channels
			c = np.mean(c, axis=2)

	# resize if needed
	if (c.shape[0], c.shape[1]) != (H, W):
		c = cv2.resize(c.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

	# clip and ensure float in [0,1]
	c = np.clip(c, 0.0, 1.0)

	# 1. generate heatmap (soft color)
	heat_uint8 = (c * 255).astype(np.uint8)
	heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
	# convert to RGB for blending
	heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

	heat_save = os.path.join(group_dir, f"{file_prefix}_heatmap.png")
	cv2.imwrite(heat_save, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

	# 2. highlight map: contributions > threshold -> yellow, else light blue
	threshold = 0.1
	highlight = np.zeros((H, W, 3), dtype=np.uint8)
	yellow = np.array([255, 255, 0], dtype=np.uint8)
	blue = np.array([0, 0, 255], dtype=np.uint8)
	highlight[c > threshold] = yellow
	highlight[c <= threshold] = blue
	high_save = os.path.join(group_dir, f"{file_prefix}_highlight_mask.png")
	cv2.imwrite(high_save, cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR))

	# 3. overlay heatmap on original (faded)
	alpha = 0.45
	orig_f = img_rgb.astype(np.float32)
	heat_f = heatmap_rgb.astype(np.float32)
	over_heat = (orig_f * (1 - alpha) + heat_f * alpha).astype(np.uint8)
	over_heat_save = os.path.join(group_dir, f"{file_prefix}_heatmap_overlay.png")
	cv2.imwrite(over_heat_save, cv2.cvtColor(over_heat, cv2.COLOR_RGB2BGR))

	# 4. overlay highlight on original
	alpha2 = 0.5
	high_f = highlight.astype(np.float32)
	over_high = (orig_f * (1 - alpha2) + high_f * alpha2).astype(np.uint8)
	over_high_save = os.path.join(group_dir, f"{file_prefix}_highlight_overlay.png")
	cv2.imwrite(over_high_save, cv2.cvtColor(over_high, cv2.COLOR_RGB2BGR))

	# done
	return


def show_erf(checkpoint_path, img_path, label_path, semantic, out_dir):
	"""
	High-level helper: compute contribution map and save visualizations.

	Steps: preprocess image -> build grad mask -> compute contrib -> save images.
	"""

	# 1. preprocess image
	img_tensor = img_process(img_path)

	# 2. grad mask
	mask = get_grad_mask(label_path, semantic)

	# 3. contribution
	contrib = get_contirb(checkpoint_path, img_tensor, grad_mask=mask)

	# 4. show and save
	show_img(img_path, contrib, out_dir)

	return


if __name__ == "__main__":

	imgs, labels = get_imgs(IMG_DIR, LABEL_DIR)

	if len(imgs) != len(labels):
		raise ValueError(f"Number of images ({len(imgs)}) does not match number of labels ({len(labels)}).\n"
						 f"Check IMG_DIR={IMG_DIR} and LABEL_DIR={LABEL_DIR} and their city subfolders.")

	for img_p, lbl_p in zip(imgs, labels):
		show_erf(checkpoint_path, img_p, lbl_p, semantic, out_dir)