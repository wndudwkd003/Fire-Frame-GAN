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
        fire_images_path = os.path.join(fire_image_path, 'train', 'images')
        fire_images = []
        for root, _, files in os.walk(fire_images_path):
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

dataset = FireDataset(train_label, fire_image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# CycleGAN 모델 초기화
input_nc_c2f = 6  # clean to fire: 6채널 입력
input_nc_f2c = 3  # fire to clean: 3채널 입력
output_nc = 3  # 3채널 출력 (화재 이미지)
model = CycleGAN(input_nc_c2f, input_nc_f2c, output_nc).to(device)

# 옵티마이저 설정
optimizer_G = optim.Adam(list(model.generator_c2f.parameters()) + list(model.generator_f2c.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_c = optim.Adam(model.discriminator_c.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_f = optim.Adam(model.discriminator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 20000
lambda_cycle = 30

# 체크포인트 로드
start_epoch = 20
if start_epoch > 0:
    checkpoint_path = os.path.join(model_result, f'checkpoint_epoch_{start_epoch}_loss_*.pth')
    checkpoints = sorted([f for f in os.listdir(model_result) if f.startswith(f'checkpoint_epoch_{start_epoch}_loss_')])
    if checkpoints:
        checkpoint_path = os.path.join(model_result, checkpoints[-1])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D_c.load_state_dict(checkpoint['optimizer_D_c_state_dict'])
        optimizer_D_f.load_state_dict(checkpoint['optimizer_D_f_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch + 1}')

# 학습 루프
for epoch in range(start_epoch, num_epochs):
    for i, (clean_image, fire_image_6ch, fire_image) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
        clean_image, fire_image_6ch, fire_image = clean_image.to(device), fire_image_6ch.to(device), fire_image.to(device)

        # 생성기 학습
        optimizer_G.zero_grad()
        
        fake_fire, recon_clean, fake_clean, recon_fire = model(fire_image_6ch, fire_image)
        
        loss_G_gan_c2f = gan_loss(model.discriminator_c(fake_fire), True)
        loss_G_gan_f2c = gan_loss(model.discriminator_f(fake_clean), True)
        
        loss_cycle_clean = cycle_consistency_loss(clean_image, recon_clean, lambda_cycle)
        loss_cycle_fire = cycle_consistency_loss(fire_image, recon_fire, lambda_cycle)
        
        loss_G = loss_G_gan_c2f + loss_G_gan_f2c + loss_cycle_clean + loss_cycle_fire
        loss_G.backward()
        optimizer_G.step()

        # 판별기 학습
        optimizer_D_c.zero_grad()
        loss_D_c_real = gan_loss(model.discriminator_c(fire_image), True)
        loss_D_c_fake = gan_loss(model.discriminator_c(fake_fire.detach()), False)
        loss_D_c = (loss_D_c_real + loss_D_c_fake) * 0.5
        loss_D_c.backward()
        optimizer_D_c.step()

        optimizer_D_f.zero_grad()
        loss_D_f_real = gan_loss(model.discriminator_f(clean_image), True)
        loss_D_f_fake = gan_loss(model.discriminator_f(fake_clean.detach()), False)
        loss_D_f = (loss_D_f_real + loss_D_f_fake) * 0.5
        loss_D_f.backward()
        optimizer_D_f.step()

        if (i + 1) % 10 == 0:
            save_comparison_images(train_result, fake_fire.detach().cpu(), fake_clean.detach().cpu(), 8, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"Loss_G: {loss_G.item():.4f}, Loss_D_c: {loss_D_c.item():.4f}, Loss_D_f: {loss_D_f.item():.4f}")

    if (epoch + 1) % 1 == 0:
        checkpoint_path = os.path.join(model_result, f'checkpoint_epoch_{epoch+1}_loss_{loss_G.item():.4f}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_c_state_dict': optimizer_D_c.state_dict(),
            'optimizer_D_f_state_dict': optimizer_D_f.state_dict(),
            'loss_G': loss_G.item(),
            'loss_D_c': loss_D_c.item(),
            'loss_D_f': loss_D_f.item(),
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

print("Training finished.")
