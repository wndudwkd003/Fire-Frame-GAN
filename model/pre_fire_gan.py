import torch
import torch.nn as nn

# ResidualBlock 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Generator 정의
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv2d(64, output_nc, kernel_size=7, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Discriminator 정의
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class CycleGAN(nn.Module):
    def __init__(self, input_nc_c2f, input_nc_f2c, output_nc):
        super(CycleGAN, self).__init__()
        self.generator_c2f = Generator(input_nc_c2f, output_nc)  # clean to fire: 6 -> 3
        self.generator_f2c = Generator(output_nc, input_nc_f2c)  # fire to clean: 3 -> 3
        self.generator_f2c_recon = Generator(input_nc_f2c, output_nc)  # 가짜 클린 이미지를 화재 이미지로 복원: 3 -> 3
        self.discriminator_c = Discriminator(output_nc)
        self.discriminator_f = Discriminator(input_nc_f2c)

    def forward(self, fire_image_6ch, fire_image):
        # fire_image_6ch: 6채널 입력 (노이즈 클린 이미지 + 화염 패치 이미지)
        # fire_image: 3채널 입력 (실제 화재 이미지)
        
        # 가짜 화재 이미지 생성
        fake_fire = self.generator_c2f(fire_image_6ch)
        # 가짜 화재 이미지를 다시 클린 이미지로 변환하여 복원
        recon_clean = self.generator_f2c(fake_fire)
        # 실제 화재 이미지를 받아서 가짜 클린 이미지 생성
        fake_clean = self.generator_f2c(fire_image)
        # 가짜 클린 이미지를 다시 화재 이미지로 변환하여 복원
        recon_fire = self.generator_f2c_recon(fake_clean)

        return fake_fire, recon_clean, fake_clean, recon_fire

def cycle_consistency_loss(real_image, recon_image, lambda_weight=10):
    return lambda_weight * nn.L1Loss()(recon_image, real_image)

def gan_loss(predicted, target_is_real):
    target = torch.ones_like(predicted) if target_is_real else torch.zeros_like(predicted)
    return nn.MSELoss()(predicted, target)
