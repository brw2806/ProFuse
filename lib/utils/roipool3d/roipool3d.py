import torch
import numpy as np

def pt_in_box3d(x, y, z, cx, bottom_y, cz, h, w,
                              l, angle, max_dis):
    cy = bottom_y - h / 2.0
    if ((abs(x - cx) > max_dis) or (abs(y - cy) > h / 2.0) or (abs(z - cz) > max_dis)):
        return 0
    
    cosa = torch.cos(angle)
    sina = torch.sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0)
    return in_flag


def roipool3d(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, 
                                  xyz, boxes3d, pts_feature, pooled_features, pooled_empty_flag):
    # // params xyz: (B, N, 3)
    # // params boxes3d: (B, M, 7)
    # // params pts_feature: (B, N, C)
    # // params pooled_features: (B, M, 512, 3+C)
    # // params pooled_empty_flag: (B, M)

    for boxes_idx in range(boxes_num):
        for i in range(batch_size):
            cnt = 0;
            for k in range(pts_num):
                cur_in_flag = pt_in_box3d(xyz[i,k,0], xyz[i,k,1], xyz[i,k,2], boxes3d[i,boxes_idx,0], boxes3d[i,boxes_idx,1], boxes3d[i,boxes_idx,2], boxes3d[i,boxes_idx,3], 
                                            boxes3d[i,boxes_idx,4], boxes3d[i,boxes_idx,5], boxes3d[i,boxes_idx,6], 10.0)
                if (cur_in_flag):
                    if (cnt < sampled_pts_num):
                        # // copy xyz
                        pooled_features[i, boxes_idx, cnt, :3] = xyz[i,k]
                        # for j in range(3):
                        #     pooled_features[i, boxes_idx, cnt, j] = xyz[i,k,j];

                        # // copy feature
                        pooled_features[i, boxes_idx, cnt, 3:] = pts_feature[i,k]
                        # import pdb; pdb.set_trace()
                        # for j in range(feature_in_len):
                        #     pooled_features[i, boxes_idx, cnt, 3 + j] = pts_feature[i,k,j];
                        cnt += 1
                    else: 
                        break
            if (cnt == 0):
                pooled_empty_flag[i, boxes_idx] = 1;
            elif (cnt < sampled_pts_num):
                iduplicate_len = sampled_pts_num // cnt
                iduplicate_idx = sampled_pts_num % cnt
                pooled_features[i, boxes_idx, :iduplicate_len*cnt] = pooled_features[i, boxes_idx, :cnt].repeat(iduplicate_len,1)
                pooled_features[i, boxes_idx, iduplicate_len*cnt:sampled_pts_num] = pooled_features[i, boxes_idx, :iduplicate_idx]
                # // duplicate same points for sampling
                # for k in range(cnt, sampled_pts_num):
                #     iduplicate_idx = k % cnt
                #     for j in range(3 + feature_in_len):
                #         pooled_features[i, boxes_idx, k, j] = pooled_features[i, boxes_idx, iduplicate_idx, j]

    return pooled_features, pooled_empty_flag