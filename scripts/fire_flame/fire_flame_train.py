import os
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
num_epochs = 300
checkpoint_dir = cfg.fire_flame_model_path
checkpoint_name = 'checkpoint_epoch_80.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded, resuming training from epoch {start_epoch}")

# 고정된 노이즈 벡터
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# 손실 로그 기록
G_losses = []
D_losses = []

# 훈련 루프
for epoch in range(start_epoch, num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        output = netD(real_data).view(-1)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_data = netG(noise)
        output = netD(fake_data.detach()).view(-1)

        errD_fake = criterion(output, fake_label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        output = netD(fake_data).view(-1)
        errG = criterion(output, real_label)
        errG.backward()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
            epoch, num_epochs, i, len(dataloader), errD.item(), errG.item()))
            fake_images = netG(fixed_noise)
            save_generated_images(fake_images, 64, epoch=epoch, idx=i)

    if epoch % 20 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'lossG': errG.item(),
            'lossD': errD.item(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
