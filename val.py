import os
import torch

from helpers.fcn_model import get_model
import helpers.config as config
from helpers.calculate_pa_miou import calculate_metrics
from helpers.calculate_saiou import cal_sa_iou
from data_sds_cityscapes import load_data 


# ======== Modified by User ========
model_paths = [
    '/root/autodl-tmp/models/_1_15_2026_BL+CSG_both__A0.0B1.0_.pth'
    ]
# ==================================



# Test On Origin CityScapes
def val(model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=config.NUM_CLASSES, checkpoint=model_path)


    print(os.path.basename(model_path))
    # print('Testing model on Origin CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

    val_iter = load_data(root=config.DATA_DIR, mode='origin', split='val')

    miou, pa = calculate_metrics(model, val_iter, device)
    miou = miou.item()
    pa = pa.item()

    # print(f"Data miou: {miou:.5f}\nPixel Accuracy: {pa:.5f}")

    return miou, pa



# Test on Foreground / Background CityScapes

FORE_NUM_CLASSES = len(config.FOREGROUND_TRAINIDS)
BACK_NUM_CLASSES = len(config.BACKGROUND_TRAINIDS)


def forebackground_val(model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=config.NUM_CLASSES, checkpoint=model_path)

    # print('Testing model on Foreground / Background CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

    fore_iter = load_data(config.DATA_DIR, mode='foreground', split='val')
    back_iter = load_data(config.DATA_DIR, mode='background', split='val')

    fore_iou, back_iou, sa_iou = cal_sa_iou(model, fore_iter, back_iter, config.FOREGROUND_TRAINIDS, config.BACKGROUND_TRAINIDS, device)

    # print(f"Foreground Data miou: {fore_iou:.5f}\nBackground Data miou: {back_iou:.5f}\nSA miou: {sa_iou}")

    return fore_iou, back_iou, sa_iou



if __name__ == "__main__" :

    for i in range(len(model_paths)):
        miou, pa = val(model_paths[i])
        fore_iou, back_iou, sa_iou = forebackground_val(model_paths[i])

        print("\n" + "=" * 50)
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

