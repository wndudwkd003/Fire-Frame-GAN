import yaml
from easydict import EasyDict


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return EasyDict(cfg)


# 설정 파일 경로
yaml_file_path = r'/home/kjy/Desktop/development/Fire-Flame-GAN/config/config.yaml'

# YAML 파일 읽기
config = read_yaml(yaml_file_path)
# fire_flame_dataset_path = config.train_dataset_path.fire_flame
# fire_flame_model_path = config.train_model_path.fire_flame

train_result_path = EasyDict()
train_result_path.pre_fire = config.train_result_path.pre_fire

model_path = EasyDict()
model_path.flame_patch = config.train_model_path.flame_patch
model_path.pre_fire = config.train_model_path.pre_fire

dataset_path = EasyDict()
dataset_path.fire_image_yolo = config.train_dataset_path.fire_flame_yolo

dataset_path.flame_patch_image = config.dataset_path.flame_patch_image
dataset_path.clean_image = config.dataset_path.clean_image
dataset_path.noisy_clean_image = config.dataset_path.noisy_clean_image
dataset_path.bounding_box = config.dataset_path.bounding_box
dataset_path.fire_image = config.dataset_path.fire_image
