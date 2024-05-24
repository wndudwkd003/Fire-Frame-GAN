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
fire_flame_dataset_path = config.train_dataset_path.fire_flame
fire_flame_model_path = config.train_model_path.fire_flame
