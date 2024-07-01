import os
import sys

# 현재 파일의 디렉토리
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 (현재 파일의 상위 디렉토리)
project_root_directory = os.path.abspath(os.path.join(current_file_directory, '../..'))

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(project_root_directory)

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.fire_flame_gan import Generator, Discriminator, weights_init
from utils.image_save_util import save_generated_images
from config.config import dataset_path, model_path

class YOLOFireFlameDataset(Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.image_paths = []
        self.bbox_list = []
        self.load_dataset()

    def load_dataset(self):
        images_dir = os.path.join(self.root, f'{self.phase}/images')
        labels_dir = os.path.join(self.root, f'{self.phase}/labels')
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                image_file = label_file.replace('.txt', '.jpg')  # Assuming images are in jpg format
                image_path = os.path.join(images_dir, image_file)
                label_path = os.path.join(labels_dir, label_file)
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    with open(label_path, 'r') as f:
                        bboxes = []
                        for line in f:
                            parts = line.strip().split()
                            # 라벨 0 (fire) 인 경우에만 추가
                            if int(parts[0]) == 0:
                                x_center, y_center, width, height = map(float, parts[1:])
                                bboxes.append([x_center, y_center, width, height])
                        self.bbox_list.append(bboxes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bboxes = self.bbox_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        patches = []
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            patch = image[y1:y2, x1:x2]
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
        return patches

def validate(netD, netG, criterion, valid_dataloader, device):
    netD.eval()
    netG.eval()
    G_losses = []
    D_losses = []
    with torch.no_grad():
        for data in valid_dataloader:
            real_data = torch.stack(data).to(device)
            batch_size = real_data.size(0)
            real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

            # Discriminator with real data
            output = netD(real_data).view(-1)
            errD_real = criterion(output, real_label)

            # Discriminator with fake data
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = netG(noise)
            output = netD(fake_data).view(-1)
            errD_fake = criterion(output, fake_label)

            errD = errD_real + errD_fake
            D_losses.append(errD.item())

            # Generator
            output = netD(fake_data).view(-1)
            errG = criterion(output, real_label)
            G_losses.append(errG.item())

    netD.train()
    netG.train()
    return sum(D_losses) / len(D_losses), sum(G_losses) / len(G_losses)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 데이터셋 및 데이터로더 설정
train_path = dataset_path.fire_image_yolo
train_dataset = YOLOFireFlameDataset(root=train_path, phase='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=lambda x: [item for sublist in x for item in sublist])

valid_path = dataset_path.fire_image_yolo
valid_dataset = YOLOFireFlameDataset(root=valid_path, phase='valid', transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=lambda x: [item for sublist in x for item in sublist])

print(f'Total number of training images in the dataset: {len(train_dataset)}')
print(f'Total number of validation images in the dataset: {len(valid_dataset)}')

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
schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.2, patience=5, verbose=True)
schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=5, verbose=True)

# 체크포인트에서 로드
start_epoch = 0
num_epochs = 3000 + 1
checkpoint_dir = model_path.pre_fire
checkpoint_name = 'checkpoint_epoch_270.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint found, resuming training from epoch {start_epoch}")

# 고정된 노이즈 벡터
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# 손실 로그 기록
G_losses = []
D_losses = []

# 훈련 루프
for epoch in range(start_epoch, num_epochs):
    for i, data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}'), 0):
        netD.zero_grad()
        real_data = torch.stack(data).to(device)
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
            epoch, num_epochs, i, len(train_dataloader), errD.item(), errG.item()))

    if epoch % 30 == 0:
        fake_images = netG(fixed_noise)
        save_generated_images(fake_images, 64, epoch=epoch)
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'lossG': errG.item(),
            'lossD': errD.item(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 검증 및 학습률 조정
    val_errD, val_errG = validate(netD, netG, criterion, valid_dataloader, device)
    print(f'Validation Loss_D: {val_errD:.4f} Validation Loss_G: {val_errG:.4f}')
    schedulerG.step(val_errG)
    schedulerD.step(val_errD)
