# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import PIL
from torchvision.utils import make_grid, save_image
import argparse
import os
import sys
sys.path.append('/home/v4r/LBW/ICCV/FeatFormer-master/')
import cv2
import torch
import numpy as np


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamapn_tracker import SiamAPNTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

import torch
 
 
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = input_tensor[:,:,(2,1,0)]
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


parser = argparse.ArgumentParser(description='siamapn tracking')
parser.add_argument('--dataset', default='UAVDark',type=str,
        help='datasets')
parser.add_argument('--dataset_root', default='/media/tj-v4r/Luck/dataset',type=str,
        help='datasets')
parser.add_argument('--config', default='./../experiments/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='./snapshot/first.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default='',action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

# load config
cfg.merge_from_file(args.config)

# cur_dir = os.path.dirname(os.path.realpath(__file__))
# dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)
dataset_root=args.dataset_root
# create model
model = ModelBuilder()

# load model
model = load_pretrain(model, args.snapshot).cuda().eval()

# build tracker
tracker = SiamAPNTracker(model)

# create dataset
dataset = DatasetFactory.create_dataset(name=args.dataset,
                                        dataset_root=dataset_root,
                                        load_img=False)

model_name = args.snapshot.split('/')[-1].split('.')[0]+str(cfg.TRACK.w1)

for v_idx, video in enumerate(dataset):
    if args.video != '':
        # test one special video
        if video.name != args.video:
            continue
    if video.name in ['DJI_0089_bus_5','DJI_0081_car_33']:#DJI_0089_bus_5 DJI_0083_car_39 #DJI_0081_car_33 'DJI_0069_person_15']:#'DJI_0069_person_15 DJI_0083_car_39']:
        
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        output_dir = '/media/tj-v4r/Luck/RAL/withoutenhanced/{}/{}'.format(args.dataset,video.name)
        # dra=['Animal4','BMX3','Horse1','Motor2','SpeedCar4','Yacht4']
        # if video.name not in dra:
        #     continue
        os.makedirs(output_dir, exist_ok=True)
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img,idx)
                pred_bbox = outputs['bbox']
                
                        #Save and show results 
                output_imname = str(idx)+'_im.jpeg'
                output_rename = str(idx)+'_re.jpeg'
                output_impath = os.path.join(output_dir, output_imname)
                output_repath = os.path.join(output_dir, output_rename)
                im=outputs['img']
                re=outputs['res']
                #save_image_tensor2cv2(im, output_impath)
                # save_image_tensor2cv2(re, output_repath)
                save_image(im, output_impath)
                save_image(re, output_repath)
                PIL.Image.open(output_impath)
                PIL.Image.open(output_repath)
                pred_bboxes.append(pred_bbox)
                
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
    
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))

# def main():
#     # load config
#     cfg.merge_from_file(args.config)

#     # cur_dir = os.path.dirname(os.path.realpath(__file__))
#     # dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)
#     dataset_root='/home/tj-v4r/Dataset/UAV123_10fps/data_seq'
#     # create model
#     model = ModelBuilder()

#     # load model
#     model = load_pretrain(model, args.snapshot).cuda().eval()

#     # build tracker
#     tracker = SiamAPNTracker(model)

#     # create dataset
#     dataset = DatasetFactory.create_dataset(name=args.dataset,
#                                             dataset_root=dataset_root,
#                                             load_img=False)

#     model_name = args.snapshot.split('/')[-1].split('.')[0]+str(cfg.TRACK.w1)

#     # OPE tracking
#     for v_idx, video in enumerate(dataset):
#         if args.video != '':
#             # test one special video
#             if video.name != args.video:
#                 continue
#         toc = 0
#         pred_bboxes = []
#         scores = []
#         track_times = []
#         for idx, (img, gt_bbox) in enumerate(video):
#             tic = cv2.getTickCount()
#             if idx == 0:
#                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
#                 gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
#                 tracker.init(img, gt_bbox_)
#                 pred_bbox = gt_bbox_
#                 scores.append(None)
#                 if 'VOT2018-LT' == args.dataset:
#                     pred_bboxes.append([1])
#                 else:
#                     pred_bboxes.append(pred_bbox)
#             else:
#                 outputs = tracker.track(img)
#                 pred_bbox = outputs['bbox']
#                 pred_bboxes.append(pred_bbox)
#                 scores.append(outputs['best_score'])
#             toc += cv2.getTickCount() - tic
#             track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
#             if idx == 0:
#                 cv2.destroyAllWindows()
#             if args.vis and idx > 0:
#                 gt_bbox = list(map(int, gt_bbox))
#                 pred_bbox = list(map(int, pred_bbox))
#                 cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
#                               (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
#                 cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
#                               (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
#                 cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                 cv2.imshow(video.name, img)
#                 cv2.waitKey(1)
#         toc /= cv2.getTickFrequency()
#         # save results

#         model_path = os.path.join('results', args.dataset, model_name)
#         if not os.path.isdir(model_path):
#             os.makedirs(model_path)
#         result_path = os.path.join(model_path, '{}.txt'.format(video.name))
#         with open(result_path, 'w') as f:
#             for x in pred_bboxes:
#                 f.write(','.join([str(i) for i in x])+'\n')
#         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
#             v_idx+1, video.name, toc, idx / toc))


# if __name__ == '__main__':
#     main()
