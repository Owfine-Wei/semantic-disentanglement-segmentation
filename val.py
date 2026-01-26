import os
import torch

import models
from configs import get_config
from helpers.calculate_pa_miou import calculate_metrics
from helpers.calculate_saiou import cal_sa_iou
from datasets.dataset_impl import load_data 


# ======== Modified by User ========
dataset_name = 'cityscapes'
model_name = 'fcn'
model_paths = [
    '/mnt/d/SemanticSegmentation/models/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth'
    ]
# ==================================

# Get config
config = get_config(dataset_name)

# Test On Origin CityScapes
def val(model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_model_function = models.get_model(model_name)
    model = get_model_function(num_classes=config.NUM_CLASSES, checkpoint=model_path)


    print(os.path.basename(model_path))
    # print('Testing model on Origin CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

    val_iter = load_data(config, mode='origin', split='val')

    miou, pa = calculate_metrics(model, val_iter, device, num_classes=config.NUM_CLASSES)
    miou = miou.item()
    pa = pa.item()

    # print(f"Data miou: {miou:.5f}\nPixel Accuracy: {pa:.5f}")

    return miou, pa



# Test on Foreground / Background CityScapes

FORE_NUM_CLASSES = len(config.FOREGROUND_TRAINIDS)
BACK_NUM_CLASSES = len(config.BACKGROUND_TRAINIDS)


def forebackground_val(model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_model_function = models.get_model(model_name)
    model = get_model_function(num_classes=config.NUM_CLASSES, checkpoint=model_path)

    # print('Testing model on Foreground / Background CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

    fore_iter = load_data(config, mode='foreground', split='val')
    back_iter = load_data(config, mode='background', split='val')

    fiou_dict, fore_iou, biou_dict, back_iou, sa_iou = cal_sa_iou(model, fore_iter, back_iter, config.FOREGROUND_TRAINIDS, config.BACKGROUND_TRAINIDS, device)

    # print(f"Foreground Data miou: {fore_iou:.5f}\nBackground Data miou: {back_iou:.5f}\nSA miou: {sa_iou}")

    return fiou_dict, fore_iou, biou_dict, back_iou, sa_iou



if __name__ == "__main__" :

    for i in range(len(model_paths)):
        miou, pa = val(model_paths[i])
        fiou_dict_id, fore_iou, biou_dict_id, back_iou, sa_iou = forebackground_val(model_paths[i])

        id_to_name = {v: k for k, v in config.TRAIN_ID_DICT.items()}

        fiou_dict_name = {id_to_name[int(k)]: v for k, v in fiou_dict_id.items()}
        biou_dict_name = {id_to_name[int(k)]: v for k, v in biou_dict_id.items()}

        print("\n" + "=" * 50)

        print(f"{'Semantic Class':<20} | {'mIoU (%)':>10}")
        print("-" * 33)
        for name, score in fiou_dict_name.items():
            print(f"{name:<20} | {score * 100:>10.2f}%")

        print("=" * 50)

        print(f"{'Semantic Class':<20} | {'mIoU (%)':>10}")
        print("-" * 33)
        for name, score in biou_dict_name.items():
            print(f"{name:<20} | {score * 100:>10.2f}%")
        
        print("=" * 50)
        print(f"{'FINAL EVALUATION SUMMARY':^50}")
        print("=" * 50)
        print(f"{'Metric':<30} | {'Value':>15}")
        print("-" * 50)
        print(f"{'Origin mIoU':<30} | {miou:>15.5f}")
        print(f"{'Origin Pixel Accuracy':<30} | {pa:>15.5f}")
        print("-" * 50)
        print(f"{'Foreground mIoU':<30} | {fore_iou:>15.5f}")
        print(f"{'Background mIoU':<30} | {back_iou:>15.5f}")
        print(f"{'SA mIoU':<30} | {sa_iou:>15.5f}")
        print("=" * 50 + "\n")

