import os
import sys
from tqdm import tqdm
from PIL import Image

# 현재 파일의 디렉토리
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 (현재 파일의 상위 디렉토리)
project_root_directory = os.path.abspath(os.path.join(current_file_directory, '../..'))

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(project_root_directory)

import csv
import cv2
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config.config import dataset_path, train_result_path, model_path
from model.pre_fire_gan import CycleGAN, cycle_consistency_loss, gan_loss
from utils.image_save_util import save_generated_images, save_comparison_images

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 원본 이미지에 랜덤 바운딩 박스처리를 하는 과정
image_select_maxnum = 100
flame_patch_image_path = dataset_path.flame_patch_image
clean_image_path = dataset_path.clean_image
noisy_clean_image_path = dataset_path.noisy_clean_image
bounding_box_path = dataset_path.bounding_box
fire_image_path = dataset_path.fire_image

train_result = train_result_path.pre_fire
model_result = model_path.pre_fire

train_label = dict()  # 노이즈처리된 이미지의 정보와 바운딩박스 정보

# 바운딩 박스 정보를 저장할 CSV 파일 경로
bounding_box_file = os.path.join(bounding_box_path, 'bounding_boxes.csv')

# CSV 파일에서 바운딩 박스 정보와 이미지를 불러와 train_label에 저장
with open(bounding_box_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        image_name = row[0]
        x, y, w, h = map(int, row[1:])
        bounding_box = (x, y, w, h)
        
        clean_image = cv2.imread(os.path.join(clean_image_path, image_name))
        noisy_clean_image = cv2.imread(os.path.join(noisy_clean_image_path, f"noisy_{image_name}.jpg"))
        flame_patch_image = cv2.imread(os.path.join(flame_patch_image_path, f"flame_patch_{image_name}.jpg"))
        
        if clean_image is None or noisy_clean_image is None or flame_patch_image is None:
            print(f"Warning: Image {image_name} not found or could not be read.")
            continue
        
        train_label[image_name] = {
            "bounding_box": bounding_box,
            "clean_image": clean_image,
            "noisy_clean_image": noisy_clean_image,
            "flame_patch_image": flame_patch_image
        }

class FireDataset(Dataset):
    def __init__(self, data_dict, fire_image_path, transform=None):
        self.data_dict = data_dict
        self.fire_image_path = fire_image_path
        self.transform = transform
        self.fire_images = self._get_fire_images(fire_image_path)

    def _get_fire_images(self, fire_image_path):
        fire_images = []
        for root, _, files in os.walk(fire_image_path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    fire_images.append(os.path.join(root, file))
        return fire_images

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        image_name = list(self.data_dict.keys())[idx]
        data = self.data_dict[image_name]
        clean_image = data['clean_image']
        noisy_clean_image = data['noisy_clean_image']
        flame_patch_image = data['flame_patch_image']

        # 실제 화재 이미지 로드
        fire_image_path = self.fire_images[idx % len(self.fire_images)]
        fire_image = cv2.imread(fire_image_path)

        if fire_image is None:
            print(f"Warning: Fire image {fire_image_path} not found or could not be read.")
            fire_image = np.zeros((clean_image.shape[0], clean_image.shape[1], 3), dtype=np.uint8)

        # numpy.ndarray를 PIL 이미지로 변환
        clean_image = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
        noisy_clean_image = Image.fromarray(cv2.cvtColor(noisy_clean_image, cv2.COLOR_BGR2RGB))
        flame_patch_image = Image.fromarray(cv2.cvtColor(flame_patch_image, cv2.COLOR_BGR2RGB))
        fire_image = Image.fromarray(cv2.cvtColor(fire_image, cv2.COLOR_BGR2RGB))

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_clean_image = self.transform(noisy_clean_image)
            flame_patch_image = self.transform(flame_patch_image)
            fire_image = self.transform(fire_image)
            fire_image_6ch = torch.cat((noisy_clean_image, flame_patch_image), dim=0)

        return clean_image, fire_image_6ch, fire_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 모든 이미지를 256x256 크기로 리사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 학습 데이터를 로드합니다
dataset = FireDataset(train_label, fire_image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# CycleGAN 모델 초기화
input_nc_c2f = 6  # clean to fire: 6채널 입력
input_nc_f2c = 3  # fire to clean: 3채널 입력
output_nc = 3  # 3채널 출력 (화재 이미지)
model = CycleGAN(input_nc_c2f, input_nc_f2c, output_nc).to(device)

# 사용자가 입력한 에포크 번호를 기반으로 체크포인트를 로드합니다
epoch = int(input("Enter the epoch number to visualize: "))
checkpoint_path = os.path.join(model_result, f'checkpoint_epoch_{epoch}_loss_*.pth')
checkpoints = sorted([f for f in os.listdir(model_result) if f.startswith(f'checkpoint_epoch_{epoch}_loss_')])

if checkpoints:
    checkpoint_path = os.path.join(model_result, checkpoints[-1])
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {checkpoint_path}')

    # 데이터 로더에서 데이터를 가져와서 결과를 시각화합니다
    for i, (clean_image, fire_image_6ch, fire_image) in enumerate(dataloader):
        clean_image, fire_image_6ch, fire_image = clean_image.to(device), fire_image_6ch.to(device), fire_image.to(device)
        model.eval()

        with torch.no_grad():
            fake_fire, recon_clean, fake_clean, recon_fire = model(fire_image_6ch, fire_image)

        # 결과 이미지를 OpenCV로 시각화합니다
        def tensor_to_cv2(tensor):
            image = tensor.cpu().numpy().transpose(1, 2, 0)
            image = (image * 0.5 + 0.5) * 255  # denormalize
            image = image.astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Original Clean Image', tensor_to_cv2(clean_image[0]))
        cv2.imshow('Generated Fire Image', tensor_to_cv2(fake_fire[0]))
        cv2.imshow('Original Fire Image', tensor_to_cv2(fire_image[0]))
        cv2.imshow('Generated Clean Image', tensor_to_cv2(fake_clean[0]))

        key = cv2.waitKey(0)
        if key == ord('q'):  # 'q' 키를 눌러 종료
            break

    cv2.destroyAllWindows()
else:
    print(f"No checkpoint found for epoch {epoch}")
