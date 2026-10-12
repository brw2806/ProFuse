import torch
import numpy as np
import os, cv2
import sys
import time
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

np.random.seed(0)
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')
pc_feature = np.load(os.path.join(save_dir, 'pc_features_np.npy'))
image_feature = np.load(os.path.join(save_dir, 'image_feature_np.npy'))

im_feat = image_feature[0, 2:5, :, :].transpose(1, 2, 0)

# plt.imshow(im_feat)

im_feat_01 = cv2.normalize(im_feat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
im_feat_uint = (im_feat_01 * 255).astype(np.uint8)
im_feat_uint = cv2.cvtColor(im_feat_uint, cv2.COLOR_BGR2RGB)
# cv2.imwrite("Image.png", im_feat_uint, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
# cv2.imshow("Image", im_feat_uint)

# im_feat_uint = cv2.imread('image-gaoqing.png', cv2.IMREAD_COLOR)

image1 = np.random.randint(2**8, size=im_feat_uint.shape, dtype=np.uint8)    #子图像1

# cv2.imshow("Image1", image1)   #显示图片
# cv2.imwrite('image1.png',image1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

image2 = im_feat_uint - image1         ##子图像2cv2.imwrite('image2.png',image2)
# cv2.imshow("Image2", image2)   #显示图片
# cv2.imwrite('image2.png', image2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#
# cv2.waitKey(0)    #等待用户操作
# cv2.destroyAllWindows()   #释放窗口



# 假设我们要计算每行相邻像素对之间的相关性
# 首先，我们需要确保图像是灰度图像
if im_feat_uint.ndim == 3:
    img_array = np.mean(im_feat_uint, axis=-1).astype(np.uint8)






