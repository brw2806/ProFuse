import torch
import numpy as np
import os, cv2
import sys
import time
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.misc


'''
分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
'''
def RGB_correlation(channel,N):
  #计算channel通道
  h,w=channel.shape
  #随机产生pixels个[0,w-1)范围内的整数序列
  row=np.random.randint(0,h-1,N)
  col=np.random.randint(0,w-1,N)
  #绘制相邻像素相关性图,统计x,y坐标
  x=[]
  h_y=[]
  v_y=[]
  d_y=[]
  for i in range(N):
    #选择当前一个像素
    x.append(channel[row[i]][col[i]])
    #水平相邻像素是它的右侧也就是同行下一列的像素
    h_y.append(channel[row[i]][col[i]+1])
    #垂直相邻像素是它的下方也就是同列下一行的像素
    v_y.append(channel[row[i]+1][col[i]])
    #对角线相邻像素是它的右下即下一行下一列的那个像素
    d_y.append(channel[row[i]+1][col[i]+1])
  #三个方向的合到一起
  x=x*3
  y=h_y+v_y+d_y

  #结果展示
  # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  # plt.scatter(x,y)
  # plt.show()

  #计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
  ex=0
  for i in range(N):
    ex+=channel[row[i]][col[i]]
  ex=ex/N
  #计算D(x)
  dx=0
  for i in range(N):
    dx+=(channel[row[i]][col[i]]-ex)**2
  dx/=N

  #水平相邻像素h_y
  #计算E(y)
  h_ey=0
  for i in range(N):
    h_ey+=channel[row[i]][col[i]+1]
  h_ey/=N
  #计算D(y)
  h_dy=0
  for i in range(N):
    h_dy+=(channel[row[i]][col[i]+1]-h_ey)**2
  h_dy/=N
  #计算协方差
  h_cov=0
  for i in range(N):
    h_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]][col[i]+1]-h_ey)
  h_cov/=N
  h_Rxy=h_cov/(np.sqrt(dx)*np.sqrt(h_dy))

  #垂直相邻像素v_y
  #计算E(y)
  v_ey=0
  for i in range(N):
    v_ey+=channel[row[i]+1][col[i]]
  v_ey/=N
  #计算D(y)
  v_dy=0
  for i in range(N):
    v_dy+=(channel[row[i]+1][col[i]]-v_ey)**2
  v_dy/=N
  #计算协方差
  v_cov=0
  for i in range(N):
    v_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]]-v_ey)
  v_cov/=N
  v_Rxy=v_cov/(np.sqrt(dx)*np.sqrt(v_dy))

  #对角线相邻像素d_y
  #计算E(y)
  d_ey=0
  for i in range(N):
    d_ey+=channel[row[i]+1][col[i]+1]
  d_ey/=N
  #计算D(y)
  d_dy=0
  for i in range(N):
    d_dy+=(channel[row[i]+1][col[i]+1]-d_ey)**2
  d_dy/=N
  #计算协方差
  d_cov=0
  for i in range(N):
    d_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]+1]-d_ey)
  d_cov/=N
  d_Rxy=d_cov/(np.sqrt(dx)*np.sqrt(d_dy))

  return h_Rxy,v_Rxy,d_Rxy,x,y

'''
分别计算图像img的各通道相邻像素的相关系数，默认随机选取3000对相邻像素
'''
def correlation(img,N=3000):
      h,w,_=img.shape
      B,G,R=cv2.split(img)
      R_Rxy=RGB_correlation(R,N)
      G_Rxy=RGB_correlation(G,N)
      B_Rxy=RGB_correlation(B,N)

      with PdfPages('output.pdf') as pdf:
          config = {
          "font.family": 'serif',
          "font.size": 12,
          "mathtext.fontset": 'stix',
          "font.serif": ['SimSun'],}

          plt.figure(figsize=(9, 3))

          #结果展示
          plt.rcParams.update(config)
          # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码

          # plt.subplot(141)
          # plt.imshow(img[:,:,(2,1,0)])
          # plt.title('原始图像')

          #子图2
          plt.subplot(131)
          plt.scatter(R_Rxy[3],R_Rxy[4],s=1,c='red')
          plt.title('通道R')
          # plt.legend(loc='upper left', title='通道R')

          #子图3
          plt.subplot(132)
          plt.scatter(G_Rxy[3],G_Rxy[4],s=1,c='green')
          plt.title('通道G')
          #子图4
          plt.subplot(133)
          plt.scatter(B_Rxy[3],B_Rxy[4],s=1,c='blue')
          plt.title('通道B')

          # 调整布局避免重叠
          plt.tight_layout()

          plt.savefig(pdf, format='pdf', dpi=300)  # dpi参数设置图像的分辨率，‌300dpi是一个常见的选择以获得较高的图像质量。‌
          plt.show()
          plt.close()
      # pdf.close()

      return R_Rxy[0:3], G_Rxy[0:3], B_Rxy[0:3]


def im_plot(raw_image, img, img1, img2):

    with PdfPages('image-share.pdf') as pdf:
          config = {
          "font.family": 'serif',
          "font.size": 12,
          "mathtext.fontset": 'stix',
          "font.serif": ['SimSun'],}

          plt.figure(figsize=(10, 6))

          #结果展示
          plt.rcParams.update(config)
          # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码

          # plt.subplot(141)
          # plt.imshow(img[:,:,(2,1,0)])
          # plt.title('原始图像')

          # 子图1
          plt.subplot(221)
          plt.imshow(raw_image[:, :, (2, 1, 0)])
          plt.title('原始图像')
          plt.xticks([])
          plt.yticks([])

          #子图2
          plt.subplot(222)
          plt.imshow(img[:,:,(2,1,0)])
          plt.title('图像特征图')
          plt.xticks([])
          plt.yticks([])

          #子图3
          plt.subplot(223)
          plt.imshow(img1[:,:,(2,1,0)])
          plt.title('特征共享一')
          plt.xticks([])
          plt.yticks([])

          #子图4
          plt.subplot(224)
          plt.imshow(img2[:,:,(2,1,0)])
          plt.title('特征共享二')
          plt.xticks([])
          plt.yticks([])

          # 调整布局避免重叠
          plt.tight_layout()

          plt.savefig(pdf, format='pdf', dpi=300)  # dpi参数设置图像的分辨率，‌300dpi是一个常见的选择以获得较高的图像质量。‌
          plt.show()
          plt.close()
    return 0


# def result_plot(i1, i6, i8, i20, p1, p6, p8, p20):
#
#     with PdfPages('output.pdf') as pdf:
#           config = {
#           "font.family": 'serif',
#           "font.size": 12,
#           "mathtext.fontset": 'stix',
#           "font.serif": ['SimSun'],}
#
#           plt.figure(figsize=(5, 6))
#
#           #结果展示
#           plt.rcParams.update(config)
#           # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
#
#           plt.subplots_adjust(hspace=0.3, wspace=0.3)
#
#           #子图1
#           plt.subplot(421)
#           plt.imshow(i1[:,:,(2,1,0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           #子图2
#           plt.subplot(422)
#           plt.imshow(p1[:,:,(2,1,0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图3
#           plt.subplot(423)
#           plt.imshow(i6[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图4
#           plt.subplot(424)
#           plt.imshow(p6[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图5
#           plt.subplot(425)
#           plt.imshow(i8[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图6
#           plt.subplot(426)
#           plt.imshow(p8[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图7
#           plt.subplot(427)
#           plt.imshow(i20[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 子图8
#           plt.subplot(428)
#           plt.imshow(p20[:, :, (2, 1, 0)])
#           plt.xticks([])
#           plt.yticks([])
#
#           # 调整布局避免重叠
#           plt.tight_layout()
#
#           plt.savefig(pdf, format='pdf', dpi=1000)  # dpi参数设置图像的分辨率，‌300dpi是一个常见的选择以获得较高的图像质量。‌
#           plt.show()
#           plt.close()
#     return 0


def result_plot(i1, i6, i8, i20, p1, p6, p8, p20):

    with PdfPages('output.pdf') as pdf:
          config = {
          "font.family": 'serif',
          "font.size": 12,
          "mathtext.fontset": 'stix',
          "font.serif": ['SimSun'],}

          # fig = plt.figure(figsize=(5, 6))   # (w, h)

          fig, axs = plt.subplots(4, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 1, 1, 1]})

          #结果展示
          plt.rcParams.update(config)
          # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码


          #子图1
          plt.subplot(421)
          plt.imshow(i1[:,:,(2,1,0)])
          plt.xticks([])
          plt.yticks([])

          #子图2
          plt.subplot(422)
          plt.imshow(p1[:,:,(2,1,0)])
          plt.xticks([])
          plt.yticks([])

          # 子图3
          plt.subplot(423)
          plt.imshow(i6[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 子图4
          plt.subplot(424)
          plt.imshow(p6[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 子图5
          plt.subplot(425)
          plt.imshow(i8[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 子图6
          plt.subplot(426)
          plt.imshow(p8[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 子图7
          plt.subplot(427)
          plt.imshow(i20[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 子图8
          plt.subplot(428)
          plt.imshow(p20[:, :, (2, 1, 0)])
          plt.xticks([])
          plt.yticks([])

          # 调整布局避免重叠
          plt.tight_layout()

          plt.savefig(pdf, format='pdf', dpi=1000)  # dpi参数设置图像的分辨率，‌300dpi是一个常见的选择以获得较高的图像质量。‌
          plt.show()
          plt.close()
    return 0


def main(image, image1, image2):
    img = image
    img1 = image1
    img2 = image2

    R_Rxy, G_Rxy, B_Rxy = correlation(img2)
    # 输出结果保留四位有效数字
    print("******该图像的各通道各方向的相关系数为*****")
    print('通道\tHorizontal\tVertical\tDiagonal')
    print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0], R_Rxy[1], R_Rxy[2]))
    print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0], G_Rxy[1], G_Rxy[2]))
    print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0], B_Rxy[1], B_Rxy[2]))


if __name__== '__main__':
    np.random.seed(0)
    raw_image = cv2.imread('000001.png', cv2.IMREAD_COLOR)
    image = cv2.imread('image_.png', cv2.IMREAD_COLOR)
    image1 = np.random.randint(2 ** 8, size=image.shape, dtype=np.uint8)  # 子图像1
    image2 = image - image1  ##子图像2cv2.imwrite('image2.png',image2)

    # main()

    im_plot(raw_image, image, image1, image2)  # 画原始图像和两份加密图像

    # i1 = cv2.imread('i1.png', cv2.IMREAD_COLOR)
    # i6 = cv2.imread('i6.png', cv2.IMREAD_COLOR)
    # i8 = cv2.imread('i8.png', cv2.IMREAD_COLOR)
    # i20 = cv2.imread('i20.png', cv2.IMREAD_COLOR)
    #
    # p1 = cv2.imread('p1_.png', cv2.IMREAD_COLOR)
    # p6 = cv2.imread('p6_.png', cv2.IMREAD_COLOR)
    # p8 = cv2.imread('p8_.png', cv2.IMREAD_COLOR)
    # p20 = cv2.imread('p20_.png', cv2.IMREAD_COLOR)
    #
    # result_plot(i1, i6, i8, i20, p1, p6, p8, p20)

    print(0)




