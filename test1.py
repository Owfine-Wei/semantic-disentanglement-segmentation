from datasets.dataset_impl import load_data
from configs import get_config
from helpers.integrated_loss import compute_integrated_loss
from models import get_model
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = get_config('cityscapes')

loader = load_data(config,'csg','train')

get_model_function = get_model('segformer')

model = get_model_function(19, None)
model = model.to(device)

with torch.no_grad():
    for images, labels, masks, origin_images, origin_labels in loader:

        criterion = nn.CrossEntropyLoss(ignore_index=255)

        images = images.to(device)
        labels = labels.to(device)
        origin_images = origin_images.to(device) 
        origin_labels = origin_labels.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.float32)

        logits_img, features_img = model(images, return_features=True, return_dict=False) 
        logits_origin_img, features_origin_img = model(origin_images, return_features=True, return_dict=False)

        loss = compute_integrated_loss(logits_img, labels, masks, logits_origin_img, origin_labels, features_img, features_origin_img, criterion, 'csg', 1, 1)
