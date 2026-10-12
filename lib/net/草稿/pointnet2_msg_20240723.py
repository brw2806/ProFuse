import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample
import copy
import numpy as np
import os

import Protocol_Plaintext as plain
import Protocol_Trivial_Sharing as cipher
import secure_layer as layer


BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


#================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()     # if i=0, IA_Layer([64, 96])
        self.ic, self.pc = channels
        rc = self.pc // 4    # if i=0, rc=24, ic=64, pc=96
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        # img_feas(1, 64, 4096), point_feas(1 96, 4096)

        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BC'N->BNC'->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)    # (BN)24
        rp = self.fc2(point_feas_f)  # (BN)24
        att = F.sigmoid(self.fc3(F.tanh(ri + rp)))     # element-wise feature add, BNx24 -> BNx1

        att = att.squeeze(1)    # BN
        att = att.view(batch, 1, -1)    # B1N  weight
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)    # img_feas(1, 64, 4096) -> img_feas_new(1, 96, 4096)
        out = img_feas_new * att

        return out   # (1, 96, 4096)


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()    # if i=0, Atten_Fusion_Conv(64, 96, 96)

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])    # if i=0, IA_Layer([64, 96])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features # (1, 96, 4096) cat (1 96, 4096) = (1 192, 4096)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))   # (1 192, 4096) -> (1 96, 4096)

        return fusion_features


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy, align_corners=True)  # (B,C,1,N)

    # xy_np = xy.cpu().numpy()
    # feature_map_np = feature_map.cpu().numpy()
    # interpolate_feature_np = grid_sample_np(feature_map_np, xy_np)
    # error = np.max(np.abs(interpolate_feature.cpu().numpy() - interpolate_feature_np))   # 1.3494050556728254e-05

    return interpolate_feature.squeeze(2)  # (B,C,N)


def get_model(input_channels = 6, use_xyz = True):
    return Pointnet2MSG(input_channels = input_channels, use_xyz = use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint = cfg.RPN.SA_CONFIG.NPOINTS[k],
                            radii = cfg.RPN.SA_CONFIG.RADIUS[k],
                            nsamples = cfg.RPN.SA_CONFIG.NSAMPLE[k],
                            mlps = mlps,
                            use_xyz = use_xyz,
                            bn = cfg.RPN.USE_BN
                    )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                if cfg.LI_FUSION.ADD_Image_Attention:   # exe if i=0, Atten_Fusion_Conv(64, 96, 96)
                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                                        cfg.LI_FUSION.POINT_CHANNELS[i]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                  kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                  stride=cfg.LI_FUSION.DeConv_Kernels[i]))

            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce), cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)


        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                    PointnetFPModule(mlp = [pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features


    def forward(self, pointcloud: torch.FloatTensor, image=None, xy=None):

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npy')

        """
        # get data input (多模态的点云特征pc_features和图像特征image_feature，以及公开的点云-图像映射矩阵xy_fps)
        image_feature = self.Img_Block[0](image)
        pc_xyz, pc_features, pc_index = self.SA_modules[0](xyz=pointcloud, features=None)    # point-wise feature extraction

        pc_index = pc_index.long().unsqueeze(-1).repeat(1, 1, 2)
        xy_copy = copy.deepcopy(xy)
        xy_fps = torch.gather(xy_copy, 1, pc_index)

        pc_features_np = pc_features.numpy()
        image_feature_np = image_feature.numpy()
        xy_fps_np = xy_fps.numpy()

        np.save(os.path.join(save_dir, 'pc_features_np.npy'), pc_features_np)
        np.save(os.path.join(save_dir, 'image_feature_np.npy'), image_feature_np)
        np.save(os.path.join(save_dir, 'xy_fps_np.npy'), xy_fps_np)
        """

        # demo_start(the 1st LI module)

        # 0. get input: pc_features(1, 96, 4096), image_feature(1, 64, 192, 640), xy_fps(1, 4096, 2)
        pc_features = np.load(os.path.join(save_dir, 'pc_features_np.npy')).astype(np.float64)
        image_feature = np.load(os.path.join(save_dir, 'image_feature_np.npy')).astype(np.float64)
        xy_fps = np.load(os.path.join(save_dir, 'xy_fps_np.npy')).astype(np.float64)

        # 1. mapping
        # 2. interpolate

        # normalize xy to [-1,1]
        xy_grid = np.zeros(xy_fps.shape)
        size_range = [1280.0, 384.0]
        xy_grid[:, :, 0] = xy_fps[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy_grid[:, :, 1] = xy_fps[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0

        img_feature_point_wise_tensor = Feature_Gather(torch.from_numpy(image_feature), torch.from_numpy(xy_grid))

        xy_grid_dim4 = xy_grid[:, np.newaxis, :, :]    # (1,4096,2)->(1,1,4096,2)

        interpolate_feature_dim4 = grid_sample_np(image_feature, xy_grid_dim4)   # (1, 64, 192, 640) -> (1,64,1,4096)

        img_feature_point_wise = interpolate_feature_dim4[:, :, -1]   # (1,64,4096)

        error = np.max(img_feature_point_wise - img_feature_point_wise_tensor.numpy())





        # 3. fusion

        fusion_features = self.Fusion_Conv[0](pc_features, img_feature_point_wise)  # FUsion -- Atten_Fusion_Conv -- IA




        # err_li_xyz = torch.max(torch.abs(li_xyz - li_xyz_0))
        # err_li_features = torch.max(torch.abs(li_features - li_features_0))
        # err_li_index = torch.max(torch.abs(li_index - li_index_0))
        # err_image = torch.max(torch.abs(image - image_0))
        # err_li_xy_cor = torch.max(torch.abs(li_xy_cor - li_xy_cor_0))
        # err_img_gather_feature = torch.max(torch.abs(img_gather_feature - img_gather_feature_0))

        # demo_end(the 1st LI module)pointnet2_msg.py


        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            img = [image]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])

            if cfg.LI_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2)
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index)    # get n' cor according index

                image = self.Img_Block[i](img[i])   # print(image.shape)
                img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))

                li_features = self.Fusion_Conv[i](li_features, img_gather_feature)  # FUsion -- Atten_Fusion_Conv -- IA

                l_xy_cor.append(li_xy_cor)
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            #for i in range(1,len(img))
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                DeConv.append(self.DeConv[i](img[i + 1]))
            de_concat = torch.cat(DeConv,dim=1)

            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return l_xyz[0], l_features[0]


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs







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