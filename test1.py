from datasets.dataset_impl import load_data
from configs import get_config
from helpers.integrated_loss import compute_integrated_loss
from models import get_model
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = get_config('cityscapes')

loader = load_data(config,'csg','train','both')

get_model_function = get_model('segformer')

model = get_model_function(19, None)
model = model.to(device)
model.eval()

for images, labels, masks, origin_images, origin_labels in loader:

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    images = images.to(device)
    labels = labels.to(device)
    origin_images = origin_images.to(device) 
    origin_labels = origin_labels.to(device, dtype=torch.long)
    masks = masks.to(device, dtype=torch.float32)

    combined_images = torch.cat([images, origin_images], dim=0)
    combined_main_out = model(combined_images)
    combined_main_out = combined_main_out.logits
    outputs_img, outputs_origin = torch.split(combined_main_out, images.size(0), dim=0)

    if outputs_img.shape[-2:] != labels.shape[-2:]:
        outputs_img = F.interpolate(outputs_img, size=labels.shape[-2:], mode='bilinear', align_corners=False)

    loss = compute_integrated_loss(outputs_img, labels, masks, outputs_origin, origin_labels, criterion, 'csg', 0.01, 1)
