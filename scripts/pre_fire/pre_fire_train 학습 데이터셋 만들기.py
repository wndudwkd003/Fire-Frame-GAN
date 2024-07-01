import os
import sys

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
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from utils.random_bounding_box import random_bounding_box
from config.config import dataset_path, model_path
from dataset.fire_flame_dataset import FireFlameDataset
from model.fire_flame_gan import Generator, Discriminator, weights_init
from utils.image_save_util import save_generated_images

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 설정
netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 체크포인트에서 로드
start_epoch = 0
num_epochs = 630
checkpoint_dir = model_path.pre_fire
checkpoint_name = f'checkpoint_epoch_{num_epochs}.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

# 화염 패치를 생성하는 함수
def generate_flame_patch(generator, device, bbox_size):
    with torch.no_grad():
        generator.eval()
        random_noise = torch.randn(1, 100, 1, 1, device=device)
        fake_image = generator(random_noise).detach().cpu().squeeze(0)
        # Numpy 배열로 변환하고 크기 조정
        fake_image = fake_image.numpy().transpose(1, 2, 0)
        fake_image = cv2.resize(fake_image, bbox_size, interpolation=cv2.INTER_LINEAR)
        # 이미지 정규화를 원래대로 되돌림
        fake_image = (fake_image * 0.5 + 0.5) * 255
        fake_image = fake_image.astype(np.uint8)
        fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
        return fake_image

# 원본 이미지에 랜덤 바운딩 박스처리를 하는 과정
image_select_maxnum = 100
flame_patch_image_path = dataset_path.flame_patch_image
clean_image_path = dataset_path.clean_image
noisy_clean_image_path = dataset_path.noisy_clean_image
bounding_box_path = dataset_path.bounding_box

train_label = dict()  # 노이즈처리된 이미지의 정보와 바운딩박스 정보

def apply_noise_image(clean_image, bounding_box):
    """
    바운딩 박스 영역에 가우시안 노이즈를 추가함
    """
    x, y, w, h = bounding_box
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, 25, (h, w, 3)).astype(np.int16)  # int16으로 변환하여 오버플로우 방지
    # 노이즈를 추가할 부분 추출
    clean_image_roi = clean_image[y:y+h, x:x+w].astype(np.int16)  # int16으로 변환하여 오버플로우 방지
    # 노이즈 추가
    noise_region = cv2.add(clean_image_roi, noise)
    noise_region = np.clip(noise_region, 0, 255).astype(np.uint8)  # 값을 클리핑하고 uint8로 변환
    # 원본 이미지에 노이즈가 추가된 부분 삽입
    noise_image = clean_image.copy()
    noise_image[y:y+h, x:x+w] = noise_region
    return noise_image

for i in range(image_select_maxnum):
    image_name =  f"hospital_{i}.jpg"
    image_path = os.path.join(clean_image_path, image_name)
    clean_image = cv2.imread(image_path)
    
    if clean_image is None:
        continue  # 이미지가 없는 경우 건너뜁니다.

    # 이미지를 받아서 랜덤한 위치에 바운딩 박스를 설정함
    bounding_box = random_bounding_box(input_image=clean_image, min_size=(64, 64), max_size=(150, 150))

    # 바운딩 박스 영역에 가우시안 노이즈 추가
    noisy_clean_image = apply_noise_image(clean_image, bounding_box)

    # 노이즈 부분으로 화염 패치 생성
    flame_patch = generate_flame_patch(generator=netG, device=device, bbox_size=(bounding_box[2], bounding_box[3]))

    # 화염 패치를 원본 이미지 크기에 맞춰 검은 배경으로 설정
    flame_patch_image = np.zeros_like(clean_image)
    flame_patch_image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = flame_patch

    # 노이즈 라벨 저장
    train_label[image_name] = {
        "bounding_box": bounding_box,
        "clean_image": clean_image,
        "noisy_clean_image": noisy_clean_image,
        "flame_patch_image": flame_patch_image,
    }

    print(f"{image_name} noisy & flame patch generator")


# 폴더 경로 생성
if not os.path.exists(noisy_clean_image_path):
    os.makedirs(noisy_clean_image_path)

if not os.path.exists(flame_patch_image_path):
    os.makedirs(flame_patch_image_path)

# bounding_box 정보를 저장할 CSV 파일 경로
bounding_box_file = os.path.join(bounding_box_path, 'bounding_boxes.csv')

# CSV 파일에 bounding_box 정보를 저장
with open(bounding_box_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'x', 'y', 'w', 'h'])  # 헤더 작성

    for image_name, data in train_label.items():
        bounding_box = data['bounding_box']
        noisy_clean_image = data['noisy_clean_image']
        flame_patch_image = data['flame_patch_image']

        # bounding_box 정보를 CSV 파일에 작성
        writer.writerow([image_name, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]])

        # noisy_clean_image 저장
        noisy_image_path = os.path.join(noisy_clean_image_path, f"noisy_{image_name}.jpg")
        cv2.imwrite(noisy_image_path, noisy_clean_image)

        # flame_patch_image 저장
        flame_image_path = os.path.join(flame_patch_image_path, f"flame_patch_{image_name}.jpg")
        cv2.imwrite(flame_image_path, flame_patch_image)

        print(f"{image_name} noisy & flame patch image save")