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

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config.config import dataset_path, train_result_path, model_path
from model.pre_fire_gan import CycleGAN, cycle_consistency_loss, gan_loss
from utils.image_save_util import save_generated_images, save_comparison_images

## 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 경로 설정 (하드코딩된 경로)
dataset_path = {
    'flame_patch_image': '/home/kjy/Desktop/development/Fire-Flame-GAN/scripts/pre_fire/flame_patch_image/flame_patch_hospital_4.jpg.jpg',
    'noisy_clean_image': '/home/kjy/Desktop/development/Fire-Flame-GAN/scripts/pre_fire/noisy_clean_image/noisy_hospital_4.jpg.jpg',
}

model_path = model_path.pre_fire

# 에포크 번호 설정
epoch = 360  # 원하는 에포크 번호로 변경
checkpoint_path = os.path.join(model_path, f'checkpoint_epoch_{epoch}_loss_*.pth')
checkpoints = sorted([f for f in os.listdir(model_path) if f.startswith(f'checkpoint_epoch_{epoch}_loss_')])

if checkpoints:
    checkpoint_path = os.path.join(model_path, checkpoints[-1])
    checkpoint = torch.load(checkpoint_path)
    model = CycleGAN(6, 3, 3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {checkpoint_path}')
else:
    print(f"No checkpoint found for epoch {epoch}")
    sys.exit()

# 데이터셋 클래스
class SimpleFireDataset(Dataset):
    def __init__(self, noisy_clean_image_path, flame_patch_image_path, transform=None):
        self.noisy_clean_image_path = noisy_clean_image_path
        self.flame_patch_image_path = flame_patch_image_path
        self.transform = transform

    def __len__(self):
        return 1  # 단일 이미지이므로 길이는 1로 설정

    def __getitem__(self, idx):
        noisy_image = cv2.imread(self.noisy_clean_image_path)
        flame_patch_image = cv2.imread(self.flame_patch_image_path)

        if noisy_image is None or flame_patch_image is None:
            raise ValueError(f"Images could not be read.")

        noisy_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        flame_patch_image = Image.fromarray(cv2.cvtColor(flame_patch_image, cv2.COLOR_BGR2RGB))

        if self.transform:
            noisy_image = self.transform(noisy_image)
            flame_patch_image = self.transform(flame_patch_image)

        combined_image = torch.cat((noisy_image, flame_patch_image), dim=0)

        return combined_image

# 트랜스폼 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 모든 이미지를 256x256 크기로 리사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 데이터셋과 데이터로더 초기화
dataset = SimpleFireDataset(dataset_path['noisy_clean_image'], dataset_path['flame_patch_image'], transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델을 평가 모드로 설정
model.eval()

# 데이터를 사용하여 모델 결과를 시각화
for idx, combined_image in enumerate(dataloader):
    combined_image = combined_image.to(device)
    
    with torch.no_grad():
        fake_fire, recon_clean, fake_clean, recon_fire = model(combined_image, combined_image[:, :3, :, :])

    # 결과 이미지를 OpenCV로 시각화
    def tensor_to_cv2(tensor):
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        image = (image * 0.5 + 0.5) * 255  # denormalize
        image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 시각화
    cv2.imshow('Noisy Clean Image', tensor_to_cv2(combined_image[0, :3, :, :]))
    cv2.imshow('Flame Patch Image', tensor_to_cv2(combined_image[0, 3:, :, :]))
    cv2.imshow('Generated Fire Image', tensor_to_cv2(fake_fire[0]))
    cv2.imshow('Generated Clean Image', tensor_to_cv2(fake_clean[0]))

    key = cv2.waitKey(0)
    if key == ord('q'):  # 'q' 키를 눌러 종료
        break

cv2.destroyAllWindows()