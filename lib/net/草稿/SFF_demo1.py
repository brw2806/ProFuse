import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample
import copy
import numpy as np
import os
import time

import Protocol_Plaintext as plain
import Protocol_Trivial_Sharing as cipher
import secure_layer as layer


def fun_mapping(xy_fps, image_feature):
    """
    image_feature(1, 64, 192, 640), xy_fps(1, 4096, 2)
    out: four corner coords (left_top, left_bottom, right_top, right_bottom) -- (1, 4096, 2)
    """
    # map to [-1,1]
    size_image = [1280.0, 384.0]  # raw image size -- (w, h)
    xy_grid = np.zeros(xy_fps.shape)  # (1,4096,2) -- (x, y)
    xy_grid[:, :, 0] = xy_fps[:, :, 0] / (size_image[0] - 1) * 2 - 1
    xy_grid[:, :, 1] = xy_fps[:, :, 1] / (size_image[1] - 1) * 2 - 1

    # scale to feature dim (w, h)
    size_image_feature = [image_feature.shape[3], image_feature.shape[2]]  # (w, h)
    xy_img = np.zeros(xy_grid.shape)  # (1,4096,2) -- (x, y)
    xy_img[:, :, 0] = ((xy_grid[:, :, 0] + 1) * (size_image_feature[0] - 1)) / 2
    xy_img[:, :, 1] = ((xy_grid[:, :, 1] + 1) * (size_image_feature[1] - 1)) / 2

    # design ################################################################
    xy_fps_cipher = cipher.share_split(xy_fps, cls="F")
    image_feature_cipher = cipher.share_split(image_feature, cls="F")

    # map to [-1,1]
    size_image = [1280, 384]  # raw image size -- (w, h)
    xy_grid_cipher = [np.zeros(xy_fps.shape).astype(np.int64) for i in range(2)]  # (1,4096,2) -- (x, y)
    for j in range(2):
        xy_grid_cipher[j][:, :, 0] = xy_fps_cipher[j][:, :, 0] / (size_image[0] - 1) * 2 - cipher.get_enlarge_int(np.array(0.5))
        xy_grid_cipher[j][:, :, 1] = xy_fps_cipher[j][:, :, 1] / (size_image[1] - 1) * 2 - cipher.get_enlarge_int(np.array(0.5))

    xy_grid_rec = cipher.share_reconstruct(xy_grid_cipher, cls='F')
    error_xy_grid = xy_grid_rec - xy_grid

    xy_grid_cipher = cipher.share_split(xy_grid, cls="F")

    # scale to feature dim (w, h)
    size_image_feature = [image_feature_cipher[0].shape[3], image_feature_cipher[0].shape[2]]  # (w, h)
    xy_img_cipher = [np.zeros(xy_grid_cipher[0].shape).astype(np.int64) for i in range(2)]     # (1,4096,2) -- (x, y)
    for j in range(2):
        xy_img_cipher[j][:, :, 0] = ((xy_grid_cipher[j][:, :, 0] + cipher.get_enlarge_int(np.array(0.5))) * (size_image_feature[0] - 1)) / 2
        xy_img_cipher[j][:, :, 1] = ((xy_grid_cipher[j][:, :, 1] + cipher.get_enlarge_int(np.array(0.5))) * (size_image_feature[1] - 1)) / 2

    xy_img_rec = cipher.share_reconstruct(xy_img_cipher, cls='F')
    error_xy_img = xy_img_rec - xy_img



    # get four corner coords
    left_top = np.zeros(xy_img.shape).astype(np.int64)  # (x0, y0)
    left_top[:, :, 0] = xy_img[:, :, 0].astype(np.int64)
    left_top[:, :, 1] = xy_img[:, :, 1].astype(np.int64)

    left_bottom = np.zeros(left_top.shape).astype(np.int64)  # (x1, y0)
    left_bottom[:, :, 0] = left_top[:, :, 0] + 1
    left_bottom[:, :, 1] = left_top[:, :, 1]

    right_top = np.zeros(left_top.shape).astype(np.int64)  # (x0, y1)
    right_top[:, :, 0] = left_top[:, :, 0]
    right_top[:, :, 1] = left_top[:, :, 1] + 1

    right_bottom = np.zeros(left_top.shape).astype(np.int64)  # (x1, y1)
    right_bottom[:, :, 0] = left_top[:, :, 0] + 1
    right_bottom[:, :, 1] = left_top[:, :, 1] + 1

    return xy_img, left_top, left_bottom, right_top, right_bottom


def fun_interpolate(image_feature, xy_img, left_top, left_bottom, right_top, right_bottom):
    """
    image_feature(1, 64, 192, 640)
    mapping point: xy_img(1, 4096, 2)
    four corner coords (left_top, left_bottom, right_top, right_bottom) -- (1, 4096, 2)
    out: point-wise image_feature(1, 64, 4096)
    """
    # get the feature of four corner coords
    left_top_feature = image_feature[:, :, left_top[:, :, 1], left_top[:, :, 0]].squeeze(2)  # (1,64,1,4096) -> (1,64,4096)
    left_bottom_feature = image_feature[:, :, left_bottom[:, :, 1], left_bottom[:, :, 0]].squeeze(2)
    right_top_feature = image_feature[:, :, right_top[:, :, 1], right_top[:, :, 0]].squeeze(2)
    right_bottom_feature = image_feature[:, :, right_bottom[:, :, 1], right_bottom[:, :, 0]].squeeze(2)

    # u_x = x - x0, u_y = y - y0, (x, y)是映射点坐标，(x0, y0)是左上角坐标，用来计算特定点的特征权重
    u_x = xy_img[:, :, 0] - left_top[:, :, 0]
    u_y = xy_img[:, :, 1] - left_top[:, :, 1]

    # 获得四个角的加权特征(1,64,4096)
    left_top_feature_w = left_top_feature * ((1 - u_x) * (1 - u_y))[:, np.newaxis, :]
    left_bottom_feature_w = left_bottom_feature * (u_x * (1 - u_y))[:, np.newaxis, :]
    right_top_feature_w = right_top_feature * ((1 - u_x) * u_y)[:, np.newaxis, :]
    right_bottom_feature_w = right_bottom_feature * (u_x * u_y)[:, np.newaxis, :]

    # 获得双线性插值后的特征(1,64,4096)
    interpolate_feature = left_top_feature_w + left_bottom_feature_w + right_top_feature_w + right_bottom_feature_w  # (1,64,4096)

    return interpolate_feature


def fun_fusion(img_feature_point_wise, pc_feature):
    """
    img_feature_point_wise(1, 64, 4096)
    pc_feature: (1, 96, 4096)
    out: point-wise fusion feature(1, 170, 4096)
    """
    batch, ch_i, num = img_feature_point_wise.shape

    img_feas_f = img_feature_point_wise.transpose(0, 2, 1).reshape(batch * num, -1)  # (1, 64, 4096) -> (1, 4096, 64) -> (4096, 64)
    point_feas_f = pc_feature.transpose(0, 2, 1).reshape(batch * num, -1)   # (1, 96, 4096) -> (1, 4096, 96) -> (4096, 96)

    # 将图像特征和点云特征映射至相同特征空间中，获得ri和rp
    fc1_w, fc1_b = np.load(os.path.join(save_dir, 'fc1_w.npy')), np.load(os.path.join(save_dir, 'fc1_b.npy'))
    fc2_w, fc2_b = np.load(os.path.join(save_dir, 'fc2_w.npy')), np.load(os.path.join(save_dir, 'fc2_b.npy'))
    ri = layer.FC_f(img_feas_f, fc1_w, bias=fc1_b)      # (4096, 64) -> (4096, 24)
    rp = layer.FC_f(point_feas_f, fc2_w, bias=fc2_b)    # (4096, 96) -> (4096, 24)

    # 图像和点云特征之和，用来计算图像特征的重要性权重
    rip = ri + rp    # learn weight (4096, 24)

    # 将特征值映射至[-1, 1]，因为特征值有正有负，且数值很大
    # 由于特征值正向很大的话，会导致指数运算结果溢出
    # 考虑Tanh函数的奇函数性质，可以将特征值转换为负数来计算指数运算，然后将相应的Tanh计算结果取相反数
    rip_neg = -np.abs(rip)
    label = np.where(rip >= 0, -1, 1)   # rip = rip_neg * label
    rip_neg_tan = layer.Tanh(rip_neg)
    rip_tan = rip_neg_tan * label       # (4096, 24)

    # 进一步对这个结果进行学习和调整
    fc3_w, fc3_b = np.load(os.path.join(save_dir, 'fc3_w.npy')), np.load(os.path.join(save_dir, 'fc3_b.npy'))
    rip_fc = layer.FC_f(rip_tan, fc3_w, bias=fc3_b)    # (4096, 24) -> (4096, 1)

    # 归一化得到每个点的权重向量(1, 1, 4096)
    att = layer.Sigmoid(rip_fc)    # map to [0, 1], get the weight, (4096, 1)

    # 获得加权后的逐点图像特征(1, 64, 4096)
    img_feas_ajust = img_feature_point_wise * att.squeeze(1).reshape(batch, 1, -1)

    # 逐点融合点云特征和调整后的图像特征(1, 96, 4096) cat (1 64, 4096) = (1 170, 4096)
    fusion_feature = np.concatenate((pc_feature, img_feas_ajust), axis=1)

    return fusion_feature


def demo(model_dir):
    """
    the 1st LI module
    """
    # get data input (多模态的点云特征pc_features和图像特征image_feature，以及公开的点云-图像映射矩阵xy_fps)
    # pc_features(1, 96, 4096), image_feature(1, 64, 192, 640), xy_fps(1, 4096, 2)
    pc_feature = np.load(os.path.join(save_dir, 'pc_features_np.npy')).astype(np.float64)
    image_feature = np.load(os.path.join(save_dir, 'image_feature_np.npy')).astype(np.float64)
    xy_fps = np.load(os.path.join(save_dir, 'xy_fps_np.npy')).astype(np.float64)

    # a = cipher.share_split(pc_feature, cls="F")
    # b = cipher.share_split(pc_feature, cls="F")


    # 1. mapping: # xy_fps -> xy_img -> four corner coords -- (1, 4096, 2)
    xy_img, left_top, left_bottom, right_top, right_bottom = fun_mapping(xy_fps, image_feature)

    # 2. interpolate: image_feature + four corner coords -> point_wise feature -- (1, 64, 4096)
    img_feature_point_wise = fun_interpolate(image_feature, xy_img, left_top, left_bottom, right_top, right_bottom)

    # 3. fusion img_feature_point_wise(1, 64, 4096) + pc_feature(1 96, 4096) -> fusion_feature(1 192, 4096)
    fusion_feature = fun_fusion(img_feature_point_wise, pc_feature)

    return fusion_feature


if __name__ == '__main__':
    np.random.seed(0)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')

    fusion_features = demo(save_dir)

    pass
