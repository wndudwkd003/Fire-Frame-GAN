import os

import sys

# 현재 파일의 디렉토리
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 (현재 파일의 상위 디렉토리)
project_root_directory = os.path.abspath(os.path.join(current_file_directory, '../..'))

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(project_root_directory)


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.fire_flame_dataset import FireFlameDataset
from model.fire_flame_gan import Generator, Discriminator, weights_init

from utils.image_save_util import save_generated_images

import config.config as cfg

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 데이터셋 및 데이터로더 설정
path = cfg.fire_flame_dataset_path
dataset = FireFlameDataset(root=path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f'Total number of images in the dataset: {len(dataset)}')

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
checkpoint_dir = cfg.fire_flame_model_path
checkpoint_name = f'checkpoint_epoch_{num_epochs}.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

# 고정된 노이즈 벡터
# fixed_noise = torch.randn(64, 100, 1, 1, device=device)


with torch.no_grad():
    for i in range(100):
        netG.eval()
        random_noise = torch.randn(1, 100, 1, 1, device=device)
        fake_image = netG(random_noise).detach().cpu().squeeze(0)
        # fake_images = netG(fixed_noise).detach().cpu()
        save_generated_images(fake_image, 64, epoch=num_epochs)