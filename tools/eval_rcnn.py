import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
import tools.train_utils.train_utils as train_utils
from lib.utils.bbox_transform import decode_bbox_target
# from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
# rom tensorboardX import SummaryWriter
import tqdm

import sys
eval_path = os.path.abspath(__file__)
tool_path = os.path.dirname(eval_path)
EPNET_PATH = os.path.dirname(tool_path)
sys.path.append(tool_path)
sys.path.append(EPNET_PATH)
sys.path.append(os.path.join(EPNET_PATH, '/lib/net'))


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format, filename = log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file = f)


def save_rpn_features(seg_result, rpn_scores_raw, pts_features, backbone_xyz, backbone_features, kitti_features_dir,
                      sample_id):
    pts_intensity = pts_features[:, 0]

    output_file = os.path.join(kitti_features_dir, '%06d.npy' % sample_id)
    xyz_file = os.path.join(kitti_features_dir, '%06d_xyz.npy' % sample_id)
    seg_file = os.path.join(kitti_features_dir, '%06d_seg.npy' % sample_id)
    intensity_file = os.path.join(kitti_features_dir, '%06d_intensity.npy' % sample_id)
    np.save(output_file, backbone_features)
    np.save(xyz_file, backbone_xyz)
    np.save(seg_file, seg_result)
    np.save(intensity_file, pts_intensity)
    rpn_scores_raw_file = os.path.join(kitti_features_dir, '%06d_rawscore.npy' % sample_id)
    np.save(rpn_scores_raw_file, rpn_scores_raw)


def eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(666)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cpu()
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cpu()
    mode = 'TEST' if args.test else 'EVAL'

    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok = True)

    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok = True)
        os.makedirs(roi_output_dir, exist_ok = True)
        os.makedirs(refine_output_dir, exist_ok = True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    dataset = dataloader.dataset
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0

    progress_bar = tqdm.tqdm(total = len(dataloader), leave = True, desc = 'eval status')

    for ind, data in enumerate(dataloader):

        if ind >= 1: break
        print(ind)

        # if ind >= 10: break
        # # if ind != 8: continue
        # print(ind)

        cnt += 1
        sample_id, pts_rect, pts_features, pts_input = \
            data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
        batch_size = len(sample_id)
        inputs = torch.from_numpy(pts_input).to(args.device).float()    # .cuda(non_blocking = True).float()

        input_data = { 'pts_input': inputs }    # 1
        # img feature
        if cfg.LI_FUSION.ENABLED:
            pts_origin_xy, img = data['pts_origin_xy'], data['img']
            pts_origin_xy = torch.from_numpy(pts_origin_xy).to(args.device).float()    # .cuda(non_blocking = True).float()
            img = torch.from_numpy(img).to(args.device).float().permute((0,3,1,2))    # .cuda(non_blocking = True).float().permute((0,3,1,2))
            input_data['pts_origin_xy'] = pts_origin_xy    # 3
            input_data['img'] = img   # 2

        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb=data['rgb']
            pts_rgb=torch.from_numpy(pts_rgb).to(args.device).float()   # .cuda(non_blocking = True).float()
            input_data['pts_rgb']=pts_rgb

        # model inference
        ret_dict = model(input_data)    # network

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        if cfg.USE_IOU_BRANCH:
            rcnn_iou_branch = ret_dict['rcnn_iou_branch'].view(batch_size, -1, ret_dict['rcnn_iou_branch'].shape[1])        ##########################TO
            rcnn_iou_branch = torch.max(rcnn_iou_branch, rcnn_iou_branch.new().resize_(rcnn_iou_branch.shape).fill_(1e-4))
            rcnn_cls =  rcnn_iou_branch * rcnn_cls

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size = anchor_size,
                                          loc_scope = cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine = True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)

            norm_scores = torch.sigmoid(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
        else:
            pred_classes = torch.argmax(rcnn_cls, dim = 1).view(-1)
            cls_norm_scores = F.softmax(rcnn_cls, dim = 1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        recalled_num = gt_num = rpn_iou = 0
        if not args.test:
            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()

            gt_boxes3d = data['gt_boxes3d']

            for k in range(batch_size):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = cur_gt_boxes3d.__len__() - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = torch.from_numpy(cur_gt_boxes3d).cuda(non_blocking = True).float()
                    iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou, _ = iou3d.max(dim = 0)
                    refined_iou, _ = iou3d.max(dim = 1)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
                    recalled_num += (gt_max_iou > 0.7).sum().item()
                    gt_num += cur_gt_boxes3d.shape[0]
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

                    # original recall
                    iou3d_in = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou_in, _ = iou3d_in.max(dim = 0)

                    for idx, thresh in enumerate(thresh_list):
                        total_roi_recalled_bbox_list[idx] += (gt_max_iou_in > thresh).sum().item()

                if not cfg.RPN.FIXED:
                    fg_mask = rpn_cls_label > 0
                    correct = ((seg_result == rpn_cls_label) & fg_mask).sum().float()
                    union = fg_mask.sum().float() + (seg_result > 0).sum().float() - correct
                    rpn_iou = correct / torch.clamp(union, min = 1.0)
                    total_rpn_iou += rpn_iou.item()

        disp_dict = { 'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox) }
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.cpu().numpy()
            pred_boxes3d_np = pred_boxes3d.cpu().numpy()
            roi_scores_raw_np = roi_scores_raw.cpu().numpy()
            raw_scores_np = raw_scores.cpu().numpy()

            rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
            seg_result_np = seg_result.cpu().numpy()
            output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                                          seg_result_np.reshape(batch_size, -1, 1)), axis = 2)

            for k in range(batch_size):
                cur_sample_id = sample_id[k]
                calib = dataset.get_calib(cur_sample_id)
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(cur_sample_id, calib, roi_boxes3d_np[k], roi_output_dir,
                                  roi_scores_raw_np[k], image_shape)
                save_kitti_format(cur_sample_id, calib, pred_boxes3d_np[k], refine_output_dir,
                                  raw_scores_np[k], image_shape)

                output_file = os.path.join(rpn_output_dir, '%06d.npy' % cur_sample_id)
                np.save(output_file, output_data.astype(np.float32))

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH
        # print('cfg.RCNN.SCORE_THRESH:',cfg.RCNN.SCORE_THRESH)
        # print('cfg.RCNN.NMS_THRESH:',cfg.RCNN.NMS_THRESH)

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx]
            pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

            cur_sample_id = sample_id[k]
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected,
                              image_shape)

    progress_bar.close()
    # dump empty files
    split_file = os.path.join(dataset.imageset_dir, '..', '..', 'ImageSets', dataset.split + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            with open(cur_file, 'w') as temp_f:
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))

    ret_dict = { 'empty_cnt': empty_cnt }

    logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
    logger.info(str(datetime.now()))

    avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(len(dataset), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rpn_iou'] = avg_rpn_iou
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                      total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = { 'Car': 0, 'Pedestrian': 1, 'Cyclist': 2 }
        ap_result_str, ap_dict = kitti_evaluate(dataset.label_dir, final_output_dir, label_split_file = split_file,
                                                current_class = name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
    return ret_dict


def eval_single_ckpt(root_result_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval')
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    # epoch_id=3
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')

    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok = True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger = logger)

    # create dataloader & network
    test_loader = create_dataloader(logger)

    if args.model_type == 'base':
        model = PointRCNN(num_classes = test_loader.dataset.num_class, use_xyz = True, mode = 'TEST')

    # model.cuda()
    model.to(args.device)

    # load checkpoint
    logger.info("==> Loading from checkpoint '{}'".format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])


    # start evaluation
    eval_one_epoch_joint(model, test_loader, epoch_id, root_result_dir, logger)   # 2


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    # DATA_PATH = os.path.join('../', 'data')
    DATA_PATH = os.path.join(EPNET_PATH, 'data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir = DATA_PATH, npoints = cfg.RPN.NUM_POINTS, split = cfg.TEST.SPLIT, mode = mode,
                                random_select = args.random_select,
                                rcnn_eval_roi_dir = args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir = args.rcnn_eval_feature_dir,
                                classes = cfg.CLASSES,
                                logger = logger)

    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, pin_memory = True,
                             num_workers = args.workers, collate_fn = test_set.collate_batch)

    return test_loader


np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description = "arg parser")
parser.add_argument('--cfg_file', type = str, default = None, help = 'specify the config for evaluation')
parser.add_argument("--eval_mode", type = str, default = 'rcnn_online', required = True, help = "specify the evaluation mode")

parser.add_argument('--eval_all', action = 'store_true', default = False, help = 'whether to evaluate all checkpoints')
parser.add_argument('--test', action = 'store_true', default = False, help = 'evaluate without ground truth')

# ckpt load model
parser.add_argument("--ckpt", type = str, default = os.path.join(tool_path, 'log/Car/models/full_epnet_with_iou_branch/ckpt/checkpoint_epoch_45.pth'), help = "specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size for evaluation')
parser.add_argument('--workers', type = int, default = 4, help = 'number of workers for dataloader')
parser.add_argument("--extra_tag", type = str, default = 'default', help = "extra tag for multiple evaluation")
parser.add_argument('--output_dir', type = str, default = os.path.join(EPNET_PATH, 'log/Car/output/full_epnet_without_iou_branch/eval_results/'), help = 'specify an output directory if needed')
parser.add_argument("--ckpt_dir", type = str, default = os.path.join(tool_path, 'log/Car/models/full_epnet_with_iou_branch/ckpt/checkpoint_epoch_45.pth'),
                    help = "specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result', action = 'store_true', default = False, help = 'save evaluation results to files')
parser.add_argument('--save_rpn_feature', action = 'store_true', default = False,
                    help = 'save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action = 'store_true', default = True,
                    help = 'sample to the same number of points')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'ignore the checkpoint smaller than this epoch')
parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
parser.add_argument("--rcnn_eval_roi_dir", type = str, default = None,
                    help = 'specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type = str, default = None,
                    help = 'specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest = 'set_cfgs', default = None, nargs = argparse.REMAINDER,
                    help = 'set extra config keys if needed')
parser.add_argument('--device', default = "cuda:0")

parser.add_argument('--model_type', type = str, default = 'base', help = 'model type')

args = parser.parse_args()

args.cfg_file = os.path.join(tool_path, 'cfgs/LI_Fusion_with_attention_use_ce_loss.yaml')
args.workers = 0
args.device = torch.device("cpu")   # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rcnn_online':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)
    print(args.ckpt)
    with torch.no_grad():
        print("hello~~~~~eval_single~~~")
        eval_single_ckpt(root_result_dir)   # 1
