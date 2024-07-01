"""
랜덤으로 바운딩 박스를 설정하는 플로그램
"""

import numpy as np


def random_bounding_box(input_image, min_size, max_size, random_state=None):
    '''
    입력 이미지를 받고, 최소, 최대 사이즈를 정하면 랜덤한 위치에 랜덤 사이즈로 원본 이미지에 바운딩 박스 데이터를 반환함
    input_image: 입력 이미지 (numpy 배열)
    min_size: 바운딩 박스의 최소 크기 (가로, 세로)
    max_size: 바운딩 박스의 최대 크기 (가로, 세로)
    random_state: 랜덤 시드 값 (기본값 None)
    
    반환값: (x, y, w, h) 형태의 바운딩 박스 좌표
    '''
    if random_state is not None:
        np.random.seed(random_state)
    
    height, width, _ = input_image.shape
    
    box_width = np.random.randint(min_size[0], max_size[0])
    box_height = np.random.randint(min_size[1], max_size[1])
    
    x = np.random.randint(0, width - box_width)
    y = np.random.randint(0, height - box_height)
    
    return (x, y, box_width, box_height)