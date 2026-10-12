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





def grid_sample_np(input, grid):
    N, C, H_in, W_in = input.shape
    N, H_out, W_out, _ = grid.shape
    output = np.random.random((N, C, H_out, W_out))
    for i in range(N):
        for j in range(C):
            for k in range(H_out):
                for l in range(W_out):
                    param = [0.0, 0.0]
                    param[0] = (W_in - 1) * (grid[i][k][l][0] + 1) / 2
                    param[1] = (H_in - 1) * (grid[i][k][l][1] + 1) / 2
                    x0 = int(param[0])
                    x1 = x0 + 1
                    y0 = int(param[1])
                    y1 = y0 + 1
                    param[0] -= x0
                    param[1] -= y0
                    left_top = input[i][j][y0][x0] * (1 - param[0]) * (1 - param[1])
                    left_bottom = input[i][j][y1][x0] * (1 - param[0]) * param[1]
                    right_top = input[i][j][y0][x1] * param[0] * (1 - param[1])
                    right_bottom = input[i][j][y1][x1] * param[0] * param[1]
                    result = left_bottom + left_top + right_bottom + right_top
                    output[i][j][k][l] = result
    return output




def demo(model_dir):
    """
    the 1st LI module
    """
    # get data input (多模态的点云特征pc_features和图像特征image_feature，以及公开的点云-图像映射矩阵xy_fps)
    # pc_features(1, 96, 4096), image_feature(1, 64, 192, 640), xy_fps(1, 4096, 2)
    pc_features = np.load(os.path.join(save_dir, 'pc_features_np.npy')).astype(np.float64)
    image_feature = np.load(os.path.join(save_dir, 'image_feature_np.npy')).astype(np.float64)
    xy_fps = np.load(os.path.join(save_dir, 'xy_fps_np.npy')).astype(np.float64)

    # 1. mapping

    # normalize xy to [-1,1]
    size_image = [1280.0, 384.0]  # raw image size -- (w, h)
    xy_grid = np.zeros(xy_fps.shape)   # (1,4096,2) -- (x, y)
    xy_grid[:, :, 0] = xy_fps[:, :, 0] / (size_image[0] - 1) * 2 - 1
    xy_grid[:, :, 1] = xy_fps[:, :, 1] / (size_image[1] - 1) * 2 - 1

    t0 = time.perf_counter()
    xy_grid_dim4 = xy_grid[:, np.newaxis, :, :]  # (1,4096,2)->(1,1,4096,2)

    interpolate_feature_dim4 = grid_sample_np(image_feature, xy_grid_dim4)  # (1, 64, 192, 640) -> (1,64,1,4096)

    img_feature_point_wise = interpolate_feature_dim4[:, :, -1]  # (1,64,4096)
    t1 = time.perf_counter()

    # design
    size_image_feature = [image_feature.shape[3], image_feature.shape[2]]   # (w, h)
    xy_img = np.zeros(xy_grid.shape)    # # (1,4096,2) -- (x, y)
    xy_img[:, :, 0] = ((xy_grid[:, :, 0] + 1) * (size_image_feature[0] - 1)) / 2
    xy_img[:, :, 1] = ((xy_grid[:, :, 1] + 1) * (size_image_feature[1] - 1)) / 2

    left_top = np.zeros(xy_img.shape).astype(np.int64)    # (x0, y0)
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

    left_top_feature = image_feature[:, :, left_top[:, :, 1], left_top[:, :, 0]].squeeze(2)
    left_bottom_feature = image_feature[:, :, left_bottom[:, :, 1], left_bottom[:, :, 0]].squeeze(2)
    right_top_feature = image_feature[:, :, right_top[:, :, 1], right_top[:, :, 0]].squeeze(2)
    right_bottom_feature = image_feature[:, :, right_bottom[:, :, 1], right_bottom[:, :, 0]].squeeze(2)

    u_x = xy_img[:, :, 0] - left_top[:, :, 0]
    u_y = xy_img[:, :, 1] - left_top[:, :, 1]

    left_top_feature_w = left_top_feature * ((1 - u_x) * (1 - u_y))[:, np.newaxis, :]
    left_bottom_feature_w = left_bottom_feature * (u_x * (1 - u_y))[:, np.newaxis, :]
    right_top_feature_w = right_top_feature * ((1 - u_x) * u_y)[:, np.newaxis, :]
    right_bottom_feature_w = right_bottom_feature * (u_x * u_y)[:, np.newaxis, :]

    interpolate_feature = left_top_feature_w + left_bottom_feature_w + right_top_feature_w + right_bottom_feature_w

    t2 = time.perf_counter()

    error_grid_sample = img_feature_point_wise - interpolate_feature
    print(t1-t0, t2-t1)

    # 2. interpolate



    # 3. fusion (input: img_feature_point_wise(1, 64, 4096), pc_features(1 96, 4096))

    batch, ch_i, num = img_feature_point_wise.shape

    img_feas_f = img_feature_point_wise.transpose(0, 2, 1).reshape(batch * num, -1)  # (1, 64, 4096) -> (1, 4096, 64) -> (4096, 64)
    point_feas_f = pc_features.transpose(0, 2, 1).reshape(batch * num, -1)   # (1, 96, 4096) -> (1, 4096, 96) -> (4096, 96)

    fc1_w, fc1_b = np.load(os.path.join(save_dir, 'fc1_w.npy')), np.load(os.path.join(save_dir, 'fc1_b.npy'))
    fc2_w, fc2_b = np.load(os.path.join(save_dir, 'fc2_w.npy')), np.load(os.path.join(save_dir, 'fc2_b.npy'))
    ri = layer.FC_f(img_feas_f, fc1_w, bias=fc1_b)      # (4096, 64) -> (4096, 24)
    rp = layer.FC_f(point_feas_f, fc2_w, bias=fc2_b)    # (4096, 96) -> (4096, 24)

    rip = ri + rp    # learn weight (4096, 24)

    rip_neg = -np.abs(rip)
    label = np.where(rip >= 0, -1, 1)   # rip = rip_neg * label
    rip_neg_tan = layer.Tanh(rip_neg)
    rip_tan = rip_neg_tan * label       # F.tanh(rip), map to [-1, 1], (4096, 24)

    fc3_w, fc3_b = np.load(os.path.join(save_dir, 'fc3_w.npy')), np.load(os.path.join(save_dir, 'fc3_b.npy'))
    rip_fc = layer.FC_f(rip_tan, fc3_w, bias=fc3_b)    # (4096, 24) -> (4096, 1)

    att = layer.Sigmoid(rip_fc)    # map to [0, 1], get the weight, (4096, 1)
    att = att.squeeze(1).reshape(batch, 1, -1)   # (4096, 1) -> (4096,) -> (1, 1, 4096)

    # conv+bn+relu
    conv1_w, conv1_b = np.load(os.path.join(save_dir, 'conv1_w.npy')), np.load(os.path.join(save_dir, 'conv1_b.npy'))
    # (1, 64, 4096) -> (1, 64, 4096, 1) -> (1, 96, 4096, 1)
    img_feas_conv = layer.Conv2d_f(img_feature_point_wise[:, :, :, np.newaxis], conv1_w[:, :, :, np.newaxis], conv1_b)

    mean, var = np.load(os.path.join(save_dir, 'bn1_mean.npy')), np.load(os.path.join(save_dir, 'bn1_var.npy'))
    gamma, beta = np.load(os.path.join(save_dir, 'bn1_w.npy')), np.load(os.path.join(save_dir, 'bn1_b.npy'))
    img_feas_bn = layer.BN2d_f(img_feas_conv, mean, var, gamma, beta)

    img_feas_new = layer.ReLU_f(img_feas_bn).squeeze(3)    # (1, 64, 4096) -> (1, 96, 4096)

    img_feas_ajust = img_feas_new * att   # point_wise image feature ajust-weight

    # point_wise feature fusion (1, 96, 4096) cat (1 96, 4096) = (1 192, 4096)
    fusion_features = np.concatenate((pc_features, img_feas_ajust), axis=1)

    # demo_end(the 1st LI module)pointnet2_msg.py

    return fusion_features


if __name__ == '__main__':
    np.random.seed(0)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')

    fusion_features = demo(save_dir)

    pass
