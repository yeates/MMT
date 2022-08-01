import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import lz4.block

import matplotlib.pyplot as plt


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
    
    
def load_parsing(img_file, h=512, w=512, c=15):
    '''
    load segmentation data, c stands for label number of segmentation
    return: array [c, h, w]
    '''
    length = h * w * c
    base = lz4.block.decompress(np.load(img_file), length)
    parsing = np.frombuffer(base, np.bool).reshape((h, w, c)).astype(np.int)
    return parsing.transpose(2, 0, 1)


def coloring_parsing(parsing, debug=False):
    '''
    visualize segmentation map
    input: one-hot segmentation data
    '''
    color_list = [
        [50,50,50], [255,192,203], [218,112,214], [220,20,60], [0,0,205], [75,0,130], \
        [176,196,222], [176,224,230], [32,178,170], [46,139,87], [250,250,210], \
        [255,228,181], [139,69,19], [255,215,0], [139,0,0], [176,224,230], [255,0,255], \
        [127,255,170], [230,230,250], [139,0,139], [47,79,79], [60,179,113]
    ]    

    h, w = parsing.shape[:2]
    cur_img = np.zeros((h,w,3)).reshape(-1, 3)
    
    for idx, img in enumerate(parsing.transpose(2, 0, 1)):
        img = img.reshape(-1, 1)
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


def save_parsing(img, file_path, file_name):
    '''
    save segmentation map into npy file
    '''
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    img = img.astype(np.bool)
    c_img = lz4.block.compress(img, store_size=False)   # compress
    np.save(f"{file_path}/{file_name}", c_img)
    

def color_edge(edge, mask):
    '''
    - edge: [0,1] \in R^[1,H,M]. Tensor
    - mask: Pixels value 0 denotes mask.
    '''
    color_settings = [[0, 0, 0], [255,140,0]]
    
    edge = edge > 0.3
    mask = mask == 0
    h, w = edge.shape[-2:]
    edge = edge.detach().cpu().squeeze().numpy()
    mask = mask.detach().cpu().squeeze().numpy()
    
    masked_edge_pos = np.where(edge * mask)
    unmasked_edge_pos = np.where(edge * (1-mask))
    
    colored_edge = np.ones((h,w,3)) * 255
    
    colored_edge[unmasked_edge_pos] = color_settings[0]
    colored_edge[masked_edge_pos] = color_settings[1]
    
    return torch.FloatTensor(colored_edge).permute(2,0,1) / 255.