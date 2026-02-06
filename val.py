import os
import torch

import argparse
import models
from configs import get_config
from helpers.calculate_pa_miou import calculate_metrics
<<<<<<< HEAD
from helpers.calculate_saiou import cal_sa_iou
from datasets.dataset_impl import load_data 


=======
from helpers.calculate_remain_iou import cal_remain_iou
from datasets.dataset_impl import load_data 



>>>>>>> master
# Get arg
parser = argparse.ArgumentParser(description='Model Validation on Origin and ForeBackground')
parser.add_argument('--dataset_name', default='', help='the origin dataset name', required=True)
parser.add_argument('--model_name', default='', help='Model type', required=True)
parser.add_argument('--model_path', default='', help='Model checkpoint (.pth / .bin file)', required=True)
arg = parser.parse_args()

# Get config
config = get_config(arg.dataset_name)

# Test On Origin CityScapes
def val():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_model_function = models.get_model(arg.model_name)
    model = get_model_function(num_classes=config.NUM_CLASSES, checkpoint=arg.model_path)


    print(os.path.basename(arg.model_path))
    # print('Testing model on Origin CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

    val_iter = load_data(config, mode='origin', split='val')

    miou_dict, miou, pa = calculate_metrics(model, val_iter, device, num_classes=config.NUM_CLASSES)

    # print(f"Data miou: {miou:.5f}\nPixel Accuracy: {pa:.5f}")

    return miou_dict, miou, pa


<<<<<<< HEAD

# Test on Foreground / Background CityScapes

FORE_NUM_CLASSES = len(config.FOREGROUND_TRAINIDS)
BACK_NUM_CLASSES = len(config.BACKGROUND_TRAINIDS)


def forebackground_val():
=======
def remain_val():
>>>>>>> master

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_model_function = models.get_model(arg.model_name)
    model = get_model_function(num_classes=config.NUM_CLASSES, checkpoint=arg.model_path)

    # print('Testing model on Foreground / Background CityScapes')
    # print(f"Using device: {device}")

    model.to(device)
    model.eval()

<<<<<<< HEAD
    fore_iter = load_data(config, mode='foreground', split='val')
    back_iter = load_data(config, mode='background', split='val')

    fiou_dict, fore_iou, biou_dict, back_iou, sa_iou = cal_sa_iou(model, fore_iter, back_iter, config.FOREGROUND_TRAINIDS, config.BACKGROUND_TRAINIDS, device)

    # print(f"Foreground Data miou: {fore_iou:.5f}\nBackground Data miou: {back_iou:.5f}\nSA miou: {sa_iou}")

    return fiou_dict, fore_iou, biou_dict, back_iou, sa_iou
=======
    flat_iter = load_data(config, mode='flat', split='val')
    construction_iter = load_data(config, mode='construction', split='val')
    object_iter = load_data(config, mode='object', split='val')
    nature_iter = load_data(config, mode='nature', split='val')
    sky_iter = load_data(config, mode='sky', split='val')
    human_iter = load_data(config, mode='human', split='val')
    vehicle_iter = load_data(config, mode='vehicle', split='val')


    flat_iou_dict, flat_iou = cal_remain_iou(model, flat_iter, config.FLAT_TRAINIDS, device)
    construction_iou_dict, construction_iou = cal_remain_iou(model, construction_iter, config.CONSTRUCTION_TRAINIDS, device)
    object_iou_dict, object_iou = cal_remain_iou(model, object_iter, config.OBJECT_TRAINIDS, device)
    nature_iou_dict, nature_iou = cal_remain_iou(model, nature_iter, config.NATURE_TRAINIDS, device)
    sky_iou_dict, sky_iou = cal_remain_iou(model, sky_iter, config.SKY_TRAINIDS, device)
    human_iou_dict, human_iou = cal_remain_iou(model, human_iter, config.HUMAN_TRAINIDS, device)
    vehicle_iou_dict, vehicle_iou = cal_remain_iou(model, vehicle_iter, config.VEHICLE_TRAINIDS, device)
    saiou = (flat_iou*2+construction_iou*3+object_iou*3+nature_iou*2+sky_iou*1+human_iou*2+vehicle_iou*6)/19

    # print(f"Foreground Data miou: {fore_iou:.5f}\nBackground Data miou: {back_iou:.5f}\nSA miou: {sa_iou}")

    return flat_iou_dict, flat_iou, construction_iou_dict, construction_iou, object_iou_dict, object_iou, nature_iou_dict, nature_iou, sky_iou_dict, sky_iou, human_iou_dict, human_iou, vehicle_iou_dict, vehicle_iou, saiou
>>>>>>> master



if __name__ == "__main__" :

    miou_dict_id, miou, pa = val()
<<<<<<< HEAD
    fiou_dict_id, fore_iou, biou_dict_id, back_iou, sa_iou = forebackground_val()
=======
    flat_iou_dict_id, flat_iou, construction_iou_dict_id, construction_iou, object_iou_dict_id, object_iou, nature_iou_dict_id, nature_iou, sky_iou_dict_id, sky_iou, human_iou_dict_id, human_iou, vehicle_iou_dict_id, vehicle_iou, saiou = remain_val()
>>>>>>> master

    id_to_name = {v: k for k, v in config.TRAIN_ID_DICT.items()}

    miou_dict_name = {id_to_name[int(k)]: v for k, v in miou_dict_id.items()}
<<<<<<< HEAD
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
=======
    flat_dict_name = {id_to_name[int(k)]: v for k, v in flat_iou_dict_id.items()}
    construction_dict_name = {id_to_name[int(k)]: v for k, v in construction_iou_dict_id.items()}
    object_dict_name = {id_to_name[int(k)]: v for k, v in object_iou_dict_id.items()}
    nature_dict_name = {id_to_name[int(k)]: v for k, v in nature_iou_dict_id.items()}
    sky_dict_name = {id_to_name[int(k)]: v for k, v in sky_iou_dict_id.items()}
    human_dict_name = {id_to_name[int(k)]: v for k, v in human_iou_dict_id.items()}
    vehicle_dict_name = {id_to_name[int(k)]: v for k, v in vehicle_iou_dict_id.items()}

    print("\n" + "=" * 50)

    # Flat
    print(f"{'Semantic Class (Flat)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in flat_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Flat':<20} | {flat_iou * 100:>10.2f}%")
    print("=" * 50)

    # Construction
    print(f"{'Semantic Class (Const)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in construction_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Const':<20} | {construction_iou * 100:>10.2f}%")
    print("=" * 50)

    # Object
    print(f"{'Semantic Class (Object)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in object_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Object':<20} | {object_iou * 100:>10.2f}%")
    print("=" * 50)

    # Nature
    print(f"{'Semantic Class (Nature)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in nature_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Nature':<20} | {nature_iou * 100:>10.2f}%")
    print("=" * 50)

    # Sky
    print(f"{'Semantic Class (Sky)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in sky_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Sky':<20} | {sky_iou * 100:>10.2f}%")
    print("=" * 50)

    # Human
    print(f"{'Semantic Class (Human)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in human_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Human':<20} | {human_iou * 100:>10.2f}%")
    print("=" * 50)

    # Vehicle
    print(f"{'Semantic Class (Vehicle)':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in vehicle_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")
    print("-" * 33)
    print(f"{'Average Vehicle':<20} | {vehicle_iou * 100:>10.2f}%")

>>>>>>> master
    
    print("=" * 50)

    print(f"{'Semantic Class':<20} | {'mIoU (%)':>10}")
    print("-" * 33)
    for name, score in miou_dict_name.items():
        print(f"{name:<20} | {score * 100:>10.2f}%")

    print("=" * 50)
    print(f"{'FINAL EVALUATION SUMMARY':^50}")
    print("=" * 50)
    print(f"{'Metric':<30} | {'Value':>15}")
    print("-" * 50)
    print(f"{'Origin mIoU':<30} | {miou:>15.5f}")
    print(f"{'Origin Pixel Accuracy':<30} | {pa:>15.5f}")
    print("-" * 50)
<<<<<<< HEAD
    print(f"{'Foreground mIoU':<30} | {fore_iou:>15.5f}")
    print(f"{'Background mIoU':<30} | {back_iou:>15.5f}")
    print(f"{'SA mIoU':<30} | {sa_iou:>15.5f}")
=======
    print(f"{'Flat mIoU':<30} | {flat_iou:>15.5f}")
    print(f"{'Construction mIoU':<30} | {construction_iou:>15.5f}")
    print(f"{'Object mIoU':<30} | {object_iou:>15.5f}")
    print(f"{'Nature mIoU':<30} | {nature_iou:>15.5f}")
    print(f"{'Sky mIoU':<30} | {sky_iou:>15.5f}")
    print(f"{'Human mIoU':<30} | {human_iou:>15.5f}")
    print(f"{'Vehicle mIoU':<30} | {vehicle_iou:>15.5f}")
    print(f"{'SA IoU':<30} | {saiou:>15.5f}")

>>>>>>> master
    print("=" * 50 + "\n")

