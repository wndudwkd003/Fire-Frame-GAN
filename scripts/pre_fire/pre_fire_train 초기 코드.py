import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.decomposition import PCA


# 현재 파일의 디렉토리
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 (현재 파일의 상위 디렉토리)
project_root_directory = os.path.abspath(os.path.join(current_file_directory, '../..'))

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(project_root_directory)

from utils.random_bounding_box import random_bounding_box
from config.config import dataset_path as cfg
from config.config import fire_flame_model_path
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
num_epochs = 200
checkpoint_dir = fire_flame_model_path
checkpoint_name = f'checkpoint_epoch_{num_epochs}.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

# 화염 패치를 생성하는 함수
def generate_flame_patch(generator, device, noisy_region):
    # 1. 노이즈 직렬화
    noisy_vector = noisy_region.flatten()

    # 2. 차원 축소 (PCA)
    pca = PCA(n_components=100)
    noisy_vector_reduced = pca.fit_transform(noisy_vector.reshape(1, -1)).flatten()

    # 3. 차원 축소된 노이즈를 텐서로 변환
    noisy_tensor = torch.tensor(noisy_vector_reduced, dtype=torch.float32).view(1, 100, 1, 1).to(device)
    
    with torch.no_grad():
        generator.eval()
        flame_patch = generator(noisy_tensor).detach().cpu().squeeze(0)
        flame_patch = transforms.ToPILImage()(flame_patch)
        flame_patch = flame_patch.resize((noisy_region.shape[1], noisy_region.shape[0]), Image.BILINEAR)
        return np.array(flame_patch)

# 원본 이미지에 랜덤 바운딩 박스처리를 하는 과정
image_select_maxnum = 30
clean_image_path = cfg.clean_image
noise_image_path = cfg.noise_image
noise_label = dict()  # 노이즈처리된 이미지의 정보와 바운딩박스 정보

def apply_homography_transform(fire_image, clean_image, bounding_box):
    """
    fire_image를 bounding_box 크기로 변환하고, clean_image의 해당 위치에 적용함
    """
    x, y, w, h = bounding_box
    # 호모그라피 행렬 계산을 위한 원본 이미지의 4개 점 (좌상단, 우상단, 우하단, 좌하단)
    src_points = np.float32([[0, 0], [fire_image.shape[1], 0], [fire_image.shape[1], fire_image.shape[0]], [0, fire_image.shape[0]]])
    # 대상 이미지의 바운딩 박스에 맞춘 4개 점
    dst_points = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    # 호모그라피 행렬 계산
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    # 호모그라피 변환 적용
    flame_patch = cv2.warpPerspective(fire_image, homography_matrix, (clean_image.shape[1], clean_image.shape[0]))
    return flame_patch

def apply_noise_image(clean_image, bounding_box):
    """
    바운딩 박스 영역에 가우시안 노이즈를 추가함
    """
    x, y, w, h = bounding_box
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, 25, (h, w, 3)).astype(np.uint8)
    # 노이즈를 추가할 부분 추출
    noisy_region = clean_image[y:y+h, x:x+w].copy()
    # 노이즈 추가
    noisy_region = cv2.add(noisy_region, noise)
    # 원본 이미지에 노이즈가 추가된 부분 삽입
    noisy_image = clean_image.copy()
    noisy_image[y:y+h, x:x+w] = noisy_region
    return noisy_image, noisy_region

for i in range(image_select_maxnum):
    image_path = os.path.join(clean_image_path, f"hospital_{i}.jpg")
    input_image = cv2.imread(image_path)
    
    if input_image is None:
        continue  # 이미지가 없는 경우 건너뜁니다.

    # 이미지를 받아서 랜덤한 위치에 바운딩 박스를 설정함
    bounding_box = random_bounding_box(input_image=input_image, min_size=(30, 30), max_size=(150, 150))

    # 바운딩 박스 영역에 가우시안 노이즈 추가
    noisy_image, noisy_region = apply_noise_image(input_image, bounding_box)

    # 노이즈 부분으로 화염 패치 생성
    flame_patch = generate_flame_patch(netG, device, noisy_region)

    # 화염 패치를 적용
    transformed_flame_patch = apply_homography_transform(flame_patch, noisy_image, bounding_box)
    combined_image = cv2.addWeighted(noisy_image, 1, transformed_flame_patch, 1, 0)

    # 결과 이미지를 화면에 표시
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 노이즈 라벨 저장
    noise_label[f"hospital_{i}.jpg"] = {
        "bounding_box": bounding_box,
        "noise_image_path": f"noisy_hospital_{i}.jpg"
    }

    # 노이즈 이미지 저장
    output_path = os.path.join(noise_image_path, f"noisy_hospital_{i}.jpg")
    cv2.imwrite(output_path, combined_image)
