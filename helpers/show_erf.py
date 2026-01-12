import torch
import torch.nn.functional as F
import helpers.config as config
import fcn_model
import cv2
import numpy as np
import os


# ======== Modified by User ========

img_path = []
label_path = []
semantic = []

out_dir = ""
checkpoint_path = ""

# ==================================



def img_process(img_path):

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

	if not isinstance(input_tensor, torch.Tensor):
		raise TypeError('input_tensor must be a torch.Tensor')

	device = input_tensor.device if input_tensor.is_cuda or input_tensor.device.type == 'cpu' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device_str = str(device)

	model = fcn_model.get_model(checkpoint=checkpoint_path, device=device_str)
	model = model.to(device)
	model.eval()

	# Prepare input with grad enabled
	input_var = input_tensor.to(device).requires_grad_(True)

	out = model(input_var)

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

	# 如果没有 grad_mask，只返回 logits
	if grad_mask is None:
		return logits

	# 处理 grad_mask -> (B,1,H,W) 或 (B,C,H,W)
	if not torch.is_tensor(grad_mask):
		grad_mask = torch.from_numpy(np.array(grad_mask))
	grad_mask = grad_mask.to(device=logits.device)
	if grad_mask.dim() == 3:
		grad_mask = grad_mask.unsqueeze(1)
	if grad_mask.dim() != 4:
		raise ValueError('grad_mask must be shape (B,1,H,W) or (B,H,W)')

	# 扩展到 logits 通道数
	if grad_mask.size(1) == 1 and logits.size(1) != 1:
		grad_for_logits = grad_mask.expand(-1, logits.size(1), -1, -1).to(dtype=logits.dtype)
	else:
		grad_for_logits = grad_mask.to(dtype=logits.dtype)

	# 使用 autograd.grad 计算输入梯度，更稳健且无需全局搜索
	input_grad = torch.autograd.grad(outputs=logits, inputs=input_var, grad_outputs=grad_for_logits, retain_graph=True, allow_unused=True)[0]

	if input_grad is None:
		raise RuntimeError('autograd.grad returned None; ensure logits depend on input and requires_grad is True on input.')

	contrib = input_grad.abs().sum(dim=1, keepdim=True)

	# 归一化到 0-1
	contrib = contrib.to(torch.float32)
	min_v = contrib.amin(dim=[1,2,3], keepdim=True)
	max_v = contrib.amax(dim=[1,2,3], keepdim=True)
	range_v = (max_v - min_v)
	eps = 1e-8
	contrib = (contrib - min_v) / (range_v + eps)

	return contrib



def show_img(img_path, contrib, out_dir):

	# 0. 读取原图并保存（RGB）
	img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Image not found: {img_path}")
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	# ensure out_dir exists
	os.makedirs(out_dir, exist_ok=True)
	base = os.path.splitext(os.path.basename(img_path))[0]
	orig_save = os.path.join(out_dir, f"{base}_orig.png")
	# save RGB but cv2.imwrite expects BGR, so convert back
	cv2.imwrite(orig_save, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

	H, W = img_rgb.shape[:2]

	# prepare contrib as HxW numpy array in 0-1
	if torch.is_tensor(contrib):
		c = contrib.detach().cpu().numpy()
	else:
		c = np.array(contrib)

	# support shapes: (1,1,H,W), (1,H,W), (H,W)
	if c.ndim == 4:
		c = c[0,0]
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

	# 1. 生成热力图 (淡一点颜色)
	heat_uint8 = (c * 255).astype(np.uint8)
	heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
	# make heatmap a bit faded by converting to float and scaling
	heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

	heat_save = os.path.join(out_dir, f"{base}_heatmap.png")
	cv2.imwrite(heat_save, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

	# 2. 生成高亮图：贡献>0.75 为黄色，其余为淡蓝
	threshold = 0.75
	highlight = np.zeros((H, W, 3), dtype=np.uint8)
	# yellow RGB
	yellow = np.array([255, 255, 0], dtype=np.uint8)
	# light blue RGB
	light_blue = np.array([173, 216, 230], dtype=np.uint8)
	highlight[c > threshold] = yellow
	highlight[c <= threshold] = light_blue
	high_save = os.path.join(out_dir, f"{base}_highlight_mask.png")
	cv2.imwrite(high_save, cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR))

	# 3. overlap heatmap and original (heatmap淡一点)
	alpha = 0.45
	orig_f = img_rgb.astype(np.float32)
	heat_f = heatmap_rgb.astype(np.float32)
	over_heat = (orig_f * (1 - alpha) + heat_f * alpha).astype(np.uint8)
	over_heat_save = os.path.join(out_dir, f"{base}_heatmap_overlay.png")
	cv2.imwrite(over_heat_save, cv2.cvtColor(over_heat, cv2.COLOR_RGB2BGR))

	# 4. overlap highlight and original
	alpha2 = 0.5
	high_f = highlight.astype(np.float32)
	over_high = (orig_f * (1 - alpha2) + high_f * alpha2).astype(np.uint8)
	over_high_save = os.path.join(out_dir, f"{base}_highlight_overlay.png")
	cv2.imwrite(over_high_save, cv2.cvtColor(over_high, cv2.COLOR_RGB2BGR))

	# 5. done (no return)
	return



def show_erf(checkpoint_path, img_path, label_path, semantic, out_dir):

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

	if not (len(img_path) == len(label_path) == len(semantic)):
		print('Error: img_path, label_path and semantic must have the same length')
	else:
		for ip, lp, sm in zip(img_path, label_path, semantic):
			show_erf(checkpoint_path, ip, lp, sm, out_dir)

