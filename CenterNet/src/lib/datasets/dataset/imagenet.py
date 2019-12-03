from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import json
import os


class ImageNet(data.Dataset):
    num_classes = 30
    default_resolution = [416, 416]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(ImageNet, self).__init__()
        if split == 'train':
            self.annot_path = 'imagenet_all_train.txt'
        else:
            self.annot_path = 'imagenet_vid_val_small.txt'
        self.max_objs = 128
        self.class_name = ['airplane', 'antelope', 'bear', 'bicycle', 'bird', 
                                   'bus', 'car', 'cattle', 'dog', 'domestic_cat', 
                                   'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 
                                   'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
                                   'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 
                                   'train', 'turtle', 'watercraft', 'whale', 'zebra']
        self.class_ids = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
                               'n01503061', 'n02924116', 'n02958343', 'n02402425',
                               'n02084071', 'n02121808', 'n02503517', 'n02118333',
                               'n02510455', 'n02342885', 'n02374451', 'n02129165',
                               'n01674464', 'n02484322', 'n03790512', 'n02324045',
                               'n02509815', 'n02411705', 'n01726692', 'n02355227',
                               'n02129604', 'n04468005', 'n01662784', 'n04530566',
                               'n02062744', 'n02391049']
        self.id2idx = {id: idx for idx, id in enumerate(self.class_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                           [-0.5832747, 0.00994535, -0.81221408],
                                           [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        self.split = split
        self.opt = opt

        print('==> initializing imagenet {} data.'.format(split))
        self.images = open(self.annot_path, 'r').read().splitlines()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out  = list(map(self._to_float, bbox[0:4]))

                    detection = {"image_id": int(image_id),
                                      "category_id": int(category_id),
                                      "bbox": bbox_out,
                                      "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                  open('{}/results.json'.format(save_dir), 'w'))
  
    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        '''coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()'''

