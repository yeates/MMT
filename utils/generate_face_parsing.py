'''
one-hot编码生成预处理parsing数据
code: yys
date: 21-4-13
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

parsing_dic = {"background":0}

def comb(cur_img, img, idx):
    cur_img[:, :, idx] += np.array(img)[:,:,0]
    return cur_img

def load_parsing(img_file, h=512, w=512, c=14):
    '''
    加载单个parsing数据, h, w, c分别为图像长、宽、分割的类别数
    '''
    length = h * w * c
    base = lz4.block.decompress(np.load(img_file), length)
    parsing = np.frombuffer(base, np.bool).reshape((h, w, c)).astype(np.int)
    return parsing

def save_parsing(img, img_name):
    '''
    存放parsing数据
    '''
    if not os.path.isdir('CelebA-HQ-Anno/'):
        os.makedirs('CelebA-HQ-Anno/')
    img = img.astype(np.bool)
    c_img = lz4.block.compress(img, store_size=False)   # 压缩数组数据
    np.save(f"CelebA-HQ-Anno/{img_name}", c_img)


def main_worker(path, pbar, c, target_size=512, debug=False):
    cur_idx, cur_img = None, None
    files = sorted(list(glob(os.path.join(path, "*.jpg"))) + list(glob(os.path.join(path, "*.png"))))
    if debug:
        files = files[:200]
        
    for cnt, file in enumerate(files):
        fname = file
        file = basename(file).split('.')[0]
        idx, pname = file.split('_')[0], file.split('_')[1:]
        pname = '_'.join(pname)
        if pname.split('_')[0] in ['r', 'l', 'u']:  # 不区分上下左右
            pname = pname.split('_')[1]
        if pname not in parsing_dic:
            parsing_dic[pname] = len(parsing_dic)
            if debug: print(len(parsing_dic), parsing_dic)
        img = Image.open(fname).resize((target_size, target_size))
        if cur_idx != idx or cnt == len(files)-1:  # 下一张人脸或者到了最后一张
            if not debug and cur_img is not None:
                save_parsing(cur_img, f"{cur_idx}")
                pbar.update(1)
            cur_idx = idx
            h,w,_ = np.array(img).shape
            cur_img = np.zeros((h,w,c))
            # print(f"Composing face parsing {idx}...")
        cur_img = comb(cur_img, img, parsing_dic[pname])


def coloring(parsing, debug=False):
    '''
    可视化parsing结果
    输入parsing是one-hot分割编码
    '''
    color_list = [
        [255,192,203], [218,112,214], [220,20,60], [0,0,205], [75,0,130], \
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
    channel = 15    # 一共有14种分割类别，加上背景为15
    pbar = tqdm(total=30000)
    path = '../../dataset_collection/CelebAMask-HQ/CelebAMask-HQ-mask-anno/'
    subdirs = os.listdir(path)
    for sd in sorted(subdirs):
        read_path = os.path.join(path, sd)
        main_worker(read_path, pbar, channel, target_size=256, debug=False)
    print(parsing_dic)