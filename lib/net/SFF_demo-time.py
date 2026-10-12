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


def fun_mapping_sec(xy_fps_cipher, w_h, save_dir):
    """
    w_h(640, 192), xy_fps(1, 4096, 2)
    out: four corner coords (left_top, left_bottom, right_top, right_bottom) -- (1, 4096, 2)
    xy_img_cipher：没有乘2^{l_D}
    """
    t0 = time.perf_counter()
    # map to [-1,1]
    size_image = [1280, 384]  # raw image size -- (w, h)
    xy_grid_cipher = [np.zeros(xy_fps_cipher[0].shape) for i in range(2)]  # (1,4096,2) -- (x, y)
    for j in range(2):
        xy_grid_cipher[j][:, :, 0] = xy_fps_cipher[j][:, :, 0] / (size_image[0] - 1) * 2 - 0.5
        xy_grid_cipher[j][:, :, 1] = xy_fps_cipher[j][:, :, 1] / (size_image[1] - 1) * 2 - 0.5

    # scale to feature dim (w, h)，(xy_img[0], xy_img[1]) -- (x, y)为映射点
    xy_img_cipher = [np.zeros(xy_grid_cipher[0].shape) for i in range(2)]     # (1,4096,2) -- (x, y)
    for j in range(2):
        xy_img_cipher[j][:, :, 0] = ((xy_grid_cipher[j][:, :, 0] + 0.5) * (w_h[0] - 1)) / 2
        xy_img_cipher[j][:, :, 1] = ((xy_grid_cipher[j][:, :, 1] + 0.5) * (w_h[1] - 1)) / 2

    t1 = time.perf_counter()

    # 获得映射点的整数部分和小数部分
    xy_img_int_cipher = [xy_img_cipher[i].astype(np.int64) for i in range(2)]   # 整数部分
    xy_img_dic_cipher = [cipher.get_enlarge_int(xy_img_cipher[i] - xy_img_int_cipher[i]) for i in range(2)]   # 小数部分（缩放后的）

    # 计算映射点所在单位区域的四个角坐标，以计算左上角(x0, y0)为基准，x0和y0分别为小于x和y的最大整数
    # 两份秘密共享分别截断后的结果之和，不一定等于秘密截断后的结果，需要对截断的小数部分进行分类讨论
    # x = xy_img_dic_cipher[0] + xy_img_dic_cipher[1]的取值区间为(-2, 2)
    # 四种情况：若x<-1, 补偿为-2；若-1<=x<0, 补偿为-1；若0<=x<1, 补偿为0（不补偿）；若x>=1, 补偿为1
    s1_cipher = cipher.SComp_v2(xy_img_dic_cipher, [cipher.get_enlarge_int(np.array(-1)), 0])
    s2_cipher = cipher.SComp_v2(xy_img_dic_cipher, [cipher.get_enlarge_int(np.array(0)), 0])
    s3_cipher = cipher.SComp_v2(xy_img_dic_cipher, [cipher.get_enlarge_int(np.array(1)), 0])
    s1 = cipher.share_reconstruct(s1_cipher, cls="I")
    s2 = cipher.share_reconstruct(s2_cipher, cls="I")
    s3 = cipher.share_reconstruct(s3_cipher, cls="I")
    # s = s1*(-2) + (1-s1)*s2*(-1) + (1-s3)*1
    s = s1 * s2 - 2 * s1 - s2 - s3 + 1    # 考虑四种情况下的补偿计算结果，公开

    # get four corner coords
    xy_img_int = cipher.share_reconstruct(xy_img_int_cipher, cls="I")  # 公开
    left_top = xy_img_int + s  # 加上这个补偿后的结果才等同于截断秘密的结果

    t2 = time.perf_counter()

    left_bottom = np.zeros(left_top.shape).astype(np.int64)  # (x1, y0)
    left_bottom[:, :, 0] = left_top[:, :, 0] + 1
    left_bottom[:, :, 1] = left_top[:, :, 1]

    right_top = np.zeros(left_top.shape).astype(np.int64)  # (x0, y1)
    right_top[:, :, 0] = left_top[:, :, 0]
    right_top[:, :, 1] = left_top[:, :, 1] + 1

    right_bottom = np.zeros(left_top.shape).astype(np.int64)  # (x1, y1)
    right_bottom[:, :, 0] = left_top[:, :, 0] + 1
    right_bottom[:, :, 1] = left_top[:, :, 1] + 1

    t3 = time.perf_counter()
    print("**********************************************************")
    print("点映射的时间：", (t1 - t0) * 1000)
    print("计算小数部分取下整的时间：", (t2 - t1) * 1000)
    print("计算四个顶点的时间：", (t3 - t2) * 1000)
    print("第一阶段总时间", (t3 - t0) * 1000)
    print("**********************************************************")

    return xy_img_cipher, left_top, left_bottom, right_top, right_bottom


def fun_interpolate_sec(image_feature_cipher, xy_img_cipher, left_top, left_bottom, right_top, right_bottom, save_dir):
    """
    image_feature(1, 64, 192, 640)
    mapping point: xy_img(1, 4096, 2)
    four corner coords (left_top, left_bottom, right_top, right_bottom) -- (1, 4096, 2)
    out: point-wise image_feature(1, 64, 4096)
    """
    t0 = time.perf_counter()
    # get the feature of four corner coords
    left_top_feature_cipher = [image_feature_cipher[i][:, :, left_top[:, :, 1], left_top[:, :, 0]].squeeze(2) for i in
                               range(2)]  # (1,64,1,4096) -> (1,64,4096)
    left_bottom_feature_cipher = [image_feature_cipher[i][:, :, left_bottom[:, :, 1], left_bottom[:, :, 0]].squeeze(2)
                                  for i in range(2)]
    right_top_feature_cipher = [image_feature_cipher[i][:, :, right_top[:, :, 1], right_top[:, :, 0]].squeeze(2) for i
                                in range(2)]
    right_bottom_feature_cipher = [
        image_feature_cipher[i][:, :, right_bottom[:, :, 1], right_bottom[:, :, 0]].squeeze(2) for i in range(2)]

    t1 = time.perf_counter()

    # u_x = x - x0, u_y = y - y0, (x, y)是映射点坐标，(x0, y0)是左上角坐标，用来计算特定点的特征权重
    u_x_cipher = [xy_img_cipher[0][:, :, 0] - left_top[:, :, 0], xy_img_cipher[1][:, :, 0]]   # u_x
    u_y_cipher = [xy_img_cipher[0][:, :, 1] - left_top[:, :, 1], xy_img_cipher[1][:, :, 1]]   # u_y
    u_x_sub_cipher = [1 - u_x_cipher[0], -u_x_cipher[1]]  # 1 - u_x
    u_y_sub_cipher = [1 - u_y_cipher[0], -u_y_cipher[1]]  # 1 - u_y
    u_x_cipher = [cipher.get_enlarge_int(u_x_cipher[i][:, np.newaxis, :]) for i in range(2)]
    u_y_cipher = [cipher.get_enlarge_int(u_y_cipher[i][:, np.newaxis, :]) for i in range(2)]
    u_x_sub_cipher = [cipher.get_enlarge_int(u_x_sub_cipher[i][:, np.newaxis, :]) for i in range(2)]
    u_y_sub_cipher = [cipher.get_enlarge_int(u_y_sub_cipher[i][:, np.newaxis, :]) for i in range(2)]

    # 获得四个角的加权特征(1,64,4096)
    left_top_feature_w_cipher = cipher.SMul(left_top_feature_cipher, u_x_sub_cipher, if_trunc='need_only')
    left_top_feature_w_cipher = cipher.SMul(left_top_feature_w_cipher, u_y_sub_cipher, if_trunc='need_only')
    left_bottom_feature_w_cipher = cipher.SMul(left_bottom_feature_cipher, u_x_cipher, if_trunc='need_only')
    left_bottom_feature_w_cipher = cipher.SMul(left_bottom_feature_w_cipher, u_y_sub_cipher, if_trunc='need_only')

    t2 = time.perf_counter()

    right_top_feature_w_cipher = cipher.SMul(right_top_feature_cipher, u_x_sub_cipher, if_trunc='need_only')
    right_top_feature_w_cipher = cipher.SMul(right_top_feature_w_cipher, u_y_cipher, if_trunc='need_only')
    right_bottom_feature_w_cipher = cipher.SMul(right_bottom_feature_cipher, u_x_cipher, if_trunc='need_only')
    right_bottom_feature_w_cipher = cipher.SMul(right_bottom_feature_w_cipher, u_y_cipher, if_trunc='need_only')

    # 获得双线性插值后的特征(1,64,4096)
    interpolate_feature_cipher = [left_top_feature_w_cipher[i] + left_bottom_feature_w_cipher[i] +
                                  right_top_feature_w_cipher[i] + right_bottom_feature_w_cipher[i] for i in
                                  range(2)]  # (1,64,4096)

    t3 = time.perf_counter()
    print("检索四个顶点特征的时间：", (t1 - t0) * 1000)
    print("计算四个顶点特征权重的时间：", (t2 - t1) * 1000)
    print("计算映射点特征的时间：", (t3 - t2) * 1000)
    print("第二阶段总时间", (t3 - t0) * 1000)
    print("**********************************************************")

    return interpolate_feature_cipher


def fun_fusion_sec(img_feature_point_wise_cipher, pc_feature_cipher, save_dir):
    """
    img_feature_point_wise(1, 64, 4096)
    pc_feature: (1, 96, 4096)
    out: point-wise fusion feature(1, 170, 4096)
    """
    t0 = time.perf_counter()
    batch, ch_i, num = img_feature_point_wise_cipher[0].shape

    img_feas_f_cipher = [img_feature_point_wise_cipher[i].transpose(0, 2, 1).reshape(batch * num, -1) for i in
                         range(2)]  # (1, 64, 4096) -> (1, 4096, 64) -> (4096, 64)
    point_feas_f_cipher = [pc_feature_cipher[i].transpose(0, 2, 1).reshape(batch * num, -1) for i in
                           range(2)]  # (1, 96, 4096) -> (1, 4096, 96) -> (4096, 96)

    fc1_w, fc1_b = np.load(os.path.join(save_dir, 'fc1_w.npy')), np.load(os.path.join(save_dir, 'fc1_b.npy'))
    fc2_w, fc2_b = np.load(os.path.join(save_dir, 'fc2_w.npy')), np.load(os.path.join(save_dir, 'fc2_b.npy'))
    fc1_w_cipher, fc1_b_cipher = cipher.share_split(fc1_w, cls="F"), cipher.share_split(fc1_b, cls="F")
    fc2_w_cipher, fc2_b_cipher = cipher.share_split(fc2_w, cls="F"), cipher.share_split(fc2_b, cls="F")
    ri_cipher = layer.SecFC(img_feas_f_cipher, fc1_w_cipher, bias=fc1_b_cipher)    # (4096, 64) -> (4096, 24)
    rp_cipher = layer.SecFC(point_feas_f_cipher, fc2_w_cipher, bias=fc2_b_cipher)  # (4096, 96) -> (4096, 24)

    # 图像和点云特征之和，用来计算图像特征的重要性权重
    rip_cipher = [ri_cipher[i] + rp_cipher[i] for i in range(2)]  # learn weight (4096, 24)

    t1 = time.perf_counter()

    # 将特征值映射至[-1, 1]，因为特征值有正有负，且数值很大
    # 由于特征值正向很大的话，会导致指数运算结果溢出
    # 考虑Tanh函数(2/(1+e^{-2x}) - 1)的奇函数性质，可以将 -2x 转换为负数来计算指数运算，然后将相应的Tanh计算结果取相反数
    rip_tan_cipher = layer.SecTanh_neg(rip_cipher)

    # 进一步对这个结果进行学习和调整，获得每个点的权重分数向量
    fc3_w, fc3_b = np.load(os.path.join(save_dir, 'fc3_w.npy')), np.load(os.path.join(save_dir, 'fc3_b.npy'))
    fc3_w_cipher, fc3_b_cipher = cipher.share_split(fc3_w, cls="F"), cipher.share_split(fc3_b, cls="F")
    rip_fc_cipher = layer.SecFC(rip_tan_cipher, fc3_w_cipher, bias=fc3_b_cipher)  # (4096, 24) -> (4096, 1)

    # 归一化得到每个点的权重
    att_cipher = layer.SecSigmoid(rip_fc_cipher)  # map to [0, 1], (4096, 1) -> (1, 1, 4096)
    att_cipher = [att_cipher[i].squeeze(1).reshape(batch, 1, -1) for i in range(2)]

    t2 = time.perf_counter()

    # 获得加权后的逐点图像特征(1, 64, 4096)
    img_feas_ajust_cipher = cipher.SMul(img_feature_point_wise_cipher, att_cipher, if_trunc='need_only')

    # 逐点融合点云特征和调整后的图像特征(1, 96, 4096) cat (1 64, 4096) = (1 170, 4096)
    fusion_feature_cipher = [np.concatenate((pc_feature_cipher[i], img_feas_ajust_cipher[i]), axis=1) for i in range(2)]

    t3 = time.perf_counter()
    print("融合特征的时间：", (t1 - t0) * 1000)
    print("计算逐点图像特征权重的时间：", (t2 - t1) * 1000)
    print("计算自适应图像特征和最终结果的时间：", (t3 - t2) * 1000)
    print("第三阶段总时间", (t3 - t0) * 1000)
    print("**********************************************************")

    return img_feas_ajust_cipher, fusion_feature_cipher, att_cipher


def SFF_demo(save_dir):
    """
    the 1st LI module
    """
    # get data input (多模态的点云特征pc_features和图像特征image_feature，以及公开的点云-图像映射矩阵xy_fps)
    # pc_features(1, 96, 4096), image_feature(1, 64, 192, 640), xy_fps(1, 4096, 2)
    pc_feature = np.load(os.path.join(save_dir, 'pc_features_np.npy'))
    image_feature = np.load(os.path.join(save_dir, 'image_feature_np.npy'))
    xy_fps = np.load(os.path.join(save_dir, 'xy_fps_np.npy'))

    # n = 4000   # 4096
    # pc_feature = pc_feature[:, :, 0: n]
    # xy_fps = xy_fps[:, 0: n, :]

    # get data ciphertext
    pc_feature_cipher = cipher.share_split(pc_feature, cls="F")         # × 2^l_D
    image_feature_cipher = cipher.share_split(image_feature, cls="F")   # × 2^l_D
    xy_fps_cipher_0 = np.random.uniform(-1280, 1280, xy_fps.shape)
    xy_fps_cipher = [xy_fps_cipher_0, xy_fps - xy_fps_cipher_0]        # 没乘2^l_D

    # 1. mapping: # xy_fps -> xy_img -> four corner coords -- (1, 4096, 2)
    _, _, h, w = image_feature.shape
    xy_img_cipher, left_top, left_bottom, right_top, right_bottom = fun_mapping_sec(xy_fps_cipher, (w, h), save_dir)

    # 2. interpolate: image_feature + four corner coords -> point_wise feature -- (1, 64, 4096)
    img_feature_point_wise_cipher = fun_interpolate_sec(image_feature_cipher, xy_img_cipher, left_top, left_bottom,
                                                        right_top, right_bottom, save_dir)

    # 3. fusion img_feature_point_wise(1, 64, 4096) + pc_feature(1 96, 4096) -> fusion_feature(1 192, 4096)
    img_feature_point_wise_cipher, fusion_feature_cipher, att_cipher = fun_fusion_sec(img_feature_point_wise_cipher,
                                                                                      pc_feature_cipher, save_dir)

    return img_feature_point_wise_cipher, fusion_feature_cipher, att_cipher


if __name__ == '__main__':
    np.random.seed(0)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')

    fusion_feature_cipher, _, att_cipher = SFF_demo(save_dir)
    fusion_feature = cipher.share_reconstruct(fusion_feature_cipher, cls="F")
    att = cipher.share_reconstruct(att_cipher, cls="F")

    # from FF_demo import FF_demo
    #
    # fusion_feature_plain, _, att_plain = FF_demo(save_dir)
    #
    # error_feature_batch_0 = (fusion_feature - fusion_feature_plain)[0, :, :]
    # error_att_batch_0 = (att - att_plain)[0, :, :]
    #
    # range = 8
    # # error_1 = np.abs(error_feature_batch_0[range: 2 * range, range: 2 * range]) * (10 ** 5)
    # error_1 = np.abs(error_feature_batch_0[0: range:, 0: range].reshape(range, range))  # (10 ** 5)
    #
    # # # hot map
    # from matplotlib.backends.backend_pdf import PdfPages
    # with PdfPages('hot_map.pdf') as pdf:
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     plt.figure(dpi=120, figsize=(range, range))
    #
    #     # config = {
    #     #     "font.family": 'serif',
    #     #     "font.size": 12,
    #     #     "mathtext.fontset": 'stix',
    #     #     "font.serif": ['SimSun'], }
    #     # plt.rcParams.update(config)
    #
    #     sns.set(font_scale=1, rc={"font.family": "Songti SC"})
    #     sns.heatmap(
    #         data=error_1,
    #         annot=True,  # 开启注释文本
    #         fmt=".1f",  # 设置每个格子中注释文本的格式，此处保留两位小数
    #         cmap=sns.diverging_palette(255, 5, as_cmap=True),   # 颜色条美化一下
    #     )
    #     # plt.xticks(rotation=90)
    #     # plt.yticks(rotation=0)
    #     plt.savefig(pdf, format='pdf', dpi=1000)  # dpi参数设置图像的分辨率，‌300dpi是一个常见的选择以获得较高的图像质量。‌
    #     plt.show()

    pass
