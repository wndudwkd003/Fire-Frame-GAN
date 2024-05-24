import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def save_generated_images(images, num_images, epoch):
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # jpg 파일의 개수를 확인
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    file_count = len(existing_files) + 1  # 현재 파일 개수에 1을 더함

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    images = vutils.make_grid(images[:num_images], padding=2, normalize=True)
    images = np.transpose(images.cpu(), (1, 2, 0))
    frame = os.path.join(output_dir, f'image_{epoch}_{file_count}.jpg')
    plt.imsave(frame, images.numpy())
    print(f'Saved generated images to {frame}')
