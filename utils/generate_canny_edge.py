'''
生成边缘预处理数据
'''
import os, sys
import cv2
import numpy as np
from tqdm import tqdm

input_img_path = './input_edge/'
output_edge_path = './output_edge/'
        
            
def canny_edge(img_path):
    m1 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    m2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    from matplotlib import pyplot as plt
    # 第一步：完成高斯平滑滤波
    img = cv2.imread(img_path,0)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img,(3,3),2)

    # 第二步：完成一阶有限差分计算，计算每一点的梯度幅值与方向
    img1 = np.zeros(img.shape,dtype="uint8") # 与原图大小相同
    theta = np.zeros(img.shape,dtype="float")  # 方向矩阵原图像大小
    img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)
    rows,cols = img.shape
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            # Gy
            Gy = (np.dot(np.array([1, 1, 1]), (m1 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            # Gx
            Gx = (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            if Gx[0] == 0:
                theta[i-1,j-1] = 90
                continue
            else:
                temp = (np.arctan(Gy[0] / Gx[0]) ) * 180 / np.pi
            if Gx[0]*Gy[0] > 0:
                if Gx[0] > 0:
                    theta[i-1,j-1] = np.abs(temp)
                else:
                    theta[i-1,j-1] = (np.abs(temp) - 180)
            if Gx[0] * Gy[0] < 0:
                if Gx[0] > 0:
                    theta[i-1,j-1] = (-1) * np.abs(temp)
                else:
                    theta[i-1,j-1] = 180 - np.abs(temp)
            img1[i-1,j-1] = (np.sqrt(Gx**2 + Gy**2))
    for i in range(1,rows - 2):
        for j in range(1, cols - 2):
            if (    ( (theta[i,j] >= -22.5) and (theta[i,j]< 22.5) ) or
                    ( (theta[i,j] <= -157.5) and (theta[i,j] >= -180) ) or
                    ( (theta[i,j] >= 157.5) and (theta[i,j] < 180) ) ):
                theta[i,j] = 0.0
            elif (    ( (theta[i,j] >= 22.5) and (theta[i,j]< 67.5) ) or
                    ( (theta[i,j] <= -112.5) and (theta[i,j] >= -157.5) ) ):
                theta[i,j] = 45.0
            elif (    ( (theta[i,j] >= 67.5) and (theta[i,j]< 112.5) ) or
                    ( (theta[i,j] <= -67.5) and (theta[i,j] >= -112.5) ) ):
                theta[i,j] = 90.0
            elif (    ( (theta[i,j] >= 112.5) and (theta[i,j]< 157.5) ) or
                    ( (theta[i,j] <= -22.5) and (theta[i,j] >= -67.5) ) ):
                theta[i,j] = -45.0

    # 第三步：进行 非极大值抑制计算
    img2 = np.zeros(img1.shape) # 非极大值抑制图像矩阵

    for i in range(1,img2.shape[0]-1):
        for j in range(1,img2.shape[1]-1):
            if (theta[i,j] == 0.0) and (img1[i,j] == np.max([img1[i,j],img1[i+1,j],img1[i-1,j]]) ):
                    img2[i,j] = img1[i,j]

            if (theta[i,j] == -45.0) and img1[i,j] == np.max([img1[i,j],img1[i-1,j-1],img1[i+1,j+1]]):
                    img2[i,j] = img1[i,j]

            if (theta[i,j] == 90.0) and  img1[i,j] == np.max([img1[i,j],img1[i,j+1],img1[i,j-1]]):
                    img2[i,j] = img1[i,j]

            if (theta[i,j] == 45.0) and img1[i,j] == np.max([img1[i,j],img1[i-1,j+1],img1[i+1,j-1]]):
                    img2[i,j] = img1[i,j]

    # 第四步：双阈值检测和边缘连接
    img3 = np.zeros(img2.shape) #定义双阈值图像
    # TL = 0.4*np.max(img2)
    # TH = 0.5*np.max(img2)
    TL = 50
    TH = 100
    #关键在这两个阈值的选择
    for i in range(1,img3.shape[0]-1): 
        for j in range(1,img3.shape[1]-1):
            if img2[i,j] < TL:
                img3[i,j] = 0
            elif img2[i,j] > TH:
                img3[i,j] = 255
            elif (( img2[i+1,j] < TH) or (img2[i-1,j] < TH )or( img2[i,j+1] < TH )or
                    (img2[i,j-1] < TH) or (img2[i-1, j-1] < TH )or ( img2[i-1, j+1] < TH) or
                    ( img2[i+1, j+1] < TH ) or ( img2[i+1, j-1] < TH) ):
                img3[i,j] = 255


    # cv2.imwrite("1.png",img)  		  # 原始图像
    # cv2.imwrite("2.png",img1)       # 梯度幅值图
    # cv2.imwrite("3.png",img2)       #非极大值抑制灰度图
    # cv2.imwrite("4.png",img3)       # 最终效果图
    # cv2.imwrite("theta.png",theta) #角度值灰度图
    # cv2.waitKey(0)
    img3 = cv2.resize(img3, (256, 256))
    return img3


if __name__ == '__main__':
    pbar = tqdm(total=30000)
    for root, dirs, files in os.walk(input_img_path):
        print('loading ' + root)
        for file in files:
            img_path = os.path.join(root, file)
            edge_img = canny_edge(img_path)
            output_path = os.path.join(output_edge_path, file)
            cv2.imwrite(output_path, edge_img)
            pbar.update(1)