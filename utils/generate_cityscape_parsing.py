'''
组合CelebAMask-HQ数据集中的单个pasring到一张图上面
code: yys
date: 3/29
'''
from glob import glob
import os
from os.path import basename
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lz4.block
import struct
import cv2

def load_parsing(img_file, h=512, w=512, c=14):
    '''
    加载单个parsing数据, h, w, c分别为图像长、宽、分割的类别数
    '''
    length = h * w * c
    base = lz4.block.decompress(np.load(img_file), length)
    parsing = np.frombuffer(base, np.bool).reshape((h, w, c)).astype(np.int)
    return parsing

def save_parsing(img, dataset_split, img_name):
    '''
    存放parsing数据
    '''
    save_path = f"CityScapes-Anno/{dataset_split}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    img = img.astype(np.bool)
    c_img = lz4.block.compress(img, store_size=False)   # 压缩数组数据
    np.save(f"{save_path}/{img_name}", c_img)


def main_worker(path, pbar, c, target_size=256, debug=False):
    searchPath = os.path.join(path, "*" , "*" , "*_gt*_labelTrainIds.png" )
    files = sorted(list(glob(searchPath)))
    if debug:
        files = files[:200]
    
    for cnt, file in enumerate(files):
        fname = basename(file).split('.')[0]
        dataset_split = os.path.dirname(file).split('/')[1]
        seg = cv2.imread(file)
        seg = cv2.resize(seg, (target_size, target_size), interpolation=cv2.INTER_NEAREST)  # 用这个函数可以防止resize影响标注的数据
        seg = np.array(seg)[:, :, 0]    # 只取一维
        seg[seg==255] = 19  
        parsing = np.eye(c)[seg.reshape(-1)].reshape((target_size, target_size, c))
        save_parsing(parsing, dataset_split, fname)
        pbar.update(1)

def coloring(parsing, debug=False):
    '''
    可视化parsing结果
    输入parsing是one-hot分割编码
    '''
    color_list = [
        [50, 50, 50], [255,192,203], [218,112,214], [220,20,60], [0,0,205], [75,0,130], \
        [176,196,222], [176,224,230], [32,178,170], [46,139,87], [250,250,210], \
        [255,228,181], [139,69,19], [255,215,0], [139,0,0], [176,224,230], [255,0,255], \
        [127,255,170], [230,230,250], [139,0,139], [47,79,79], [60,179,113]
    ]    

    h, w = parsing.shape[:2]
    cur_img = np.zeros((h,w,3)).reshape(-1, 3)
    
    for idx, img in enumerate(parsing.transpose(2, 0, 1)):
        img = img.reshape(-1, 1)
        # 取差集合
        new_parsing_pos = set(np.argwhere(img)[:, 0])
        cur_parsing_pos = set(np.argwhere(cur_img)[:, 0])
        pos = list(new_parsing_pos - cur_parsing_pos)

        cur_img[pos] = color_list[idx]
    
    colored_img = Image.fromarray(np.uint8(cur_img).reshape(h, w, 3))
    if debug:
        plt.figure(f"{idx}")
        plt.imshow(colored_img)
        plt.show()
    return colored_img


if __name__ == '__main__':
    channel = 20    # 包含背景、void一共有20种分割类别
    pbar = tqdm(total=5000)
    path = 'gtFine/'
    main_worker(path, pbar, channel, target_size=512, debug=False)