import os
import sys
import time
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from trackers.kcftracker import *
from trackers.siamfc import *

from my_utils import *
from A3C import *
from utils_A3C import *

#from yolov3.models import *
#from yolov3.utils.utils import *
#from yolov3.utils.datasets import *

sys.path.append('pytorch_ssd/')
from pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

sys.path.append('CenterNet/src/lib')
from CenterNet.src.lib.detectors.detector_factory import detector_factory
from opts import opts


class Detector(nn.Module):
    
    def __init__(self, device, model='yolov3', n_classes=80, img_size=416):
        super(Detector, self).__init__()
        self.device = device
        self.model = model
        self.n_classes = n_classes
        self.img_size = img_size
        
        if model == 'yolov3':
            if n_classes == 80:
                self.detector = Darknet("yolov3/cfg/yolov3-spp.cfg", img_size=self.img_size).to(self.device)
                load_darknet_weights(self.detector, "yolov3/weights/yolov3-spp.weights")
            elif n_classes == 30:
                self.detector = Darknet("yolov3/cfg/yolov3-spp-vid.cfg", img_size=self.img_size).to(self.device)
                chkpt = torch.load('yolov3/weights/last_retrain.pt', map_location=self.device)
                self.detector.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1}, strict=False)
            self.detector.eval()
            
        elif model == 'mb2-ssd':
            if n_classes == 80:
                self.detector = None
            elif n_classes == 30:
                net = create_mobilenetv2_ssd_lite(n_classes+1, width_mult=1, is_test=True)
                net.load('pytorch_ssd/models/mb2-ssd-lite-Epoch-60-Loss-64.678246_retrain.pth')
                self.detector = create_mobilenetv2_ssd_lite_predictor(net, nms_method='hard', device=self.device)
            
        elif model == 'ctnet':
            if n_classes == 80:
                model_path = 'CenterNet/models/ctdet_coco_resdcn18.pth'
                opt = opts().init('{} --test --load_model {} --arch {} --head_conv {}'.format('ctdet', model_path, 'resdcn_18', 64).split(' '))
            elif n_classes == 30:
                model_path = 'CenterNet/exp/ctdet/imagenet/model_best.pth'
                #opt = opts().init('{} --test --dataset imagenet --load_model {} --arch {} --head_conv {}'.format('ctdet', model_path, 'resdcn_18', 64).split(' '))
                opt = opts().init('{} --test --dataset imagenet --load_model {} --arch {} --head_conv {} --flip_test'.format('ctdet', model_path, 'resdcn_18', 64).split(' '))
                opt.num_classes = n_classes
                opt.heads = {'hm': n_classes, 'wh': 2, 'reg': 2}
            opt.input_h, opt.input_w = self.img_size, self.img_size
            self.detector = detector_factory[opt.task](opt)
            
    def detect(self, img, conf_thres=0.3, nms_thres=0.5):
        boxes = []
        if self.model == 'yolov3':
            img_ = letterbox(img, new_shape=self.img_size)[0]
            img_ = img_[:, :, ::-1].transpose(2, 0, 1)
            img_ = np.ascontiguousarray(img_, dtype=np.float32)
            img_ = torch.from_numpy(img_/255.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred, _ = self.detector(img_)
                detections = my_nms(pred, conf_thres, nms_thres)[0]
                if detections is not None:   
                    detections[:, :4] = scale_coords(img_.shape[2:], detections[:, :4], img.shape).round()
                    boxes = [det.data.cpu().numpy() for det in detections]
                    
        elif self.model == 'mb2-ssd':
            bbox, labels, probs = self.detector.predict(img)
            bbox = bbox.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            probs = probs.data.cpu().numpy()
            for i in range(bbox.shape[0]):
                if probs[i] >= conf_thres:
                    cls_probs = np.zeros((self.n_classes,1))
                    cls_probs[labels[i]-1] = 1
                    boxes.append(np.concatenate((bbox[i], probs[i], cls_probs, labels[i]-1), axis=None))
            boxes.sort(key=lambda x: -x[4])
                    
        elif self.model == 'ctnet':
            ret = self.detector.run(img)['results']
            for i in range(1, self.n_classes+1):
                for bbox in ret[i]:
                    if bbox[4] >= conf_thres:
                        cls_probs = np.zeros((self.n_classes,1))
                        cls_probs[i-1] = 1
                        boxes.append(np.concatenate((bbox, cls_probs, [i-1]), axis=None))
            # boxes.sort(key=lambda x: -x[4])
        return boxes
    


class Scheduler(nn.Module):
    
    def __init__(self, device, phase, model, base_interval):
        super(Scheduler, self).__init__()
        self.device = device
        self.phase = phase
        self.model = model
        self.base_interval = base_interval
        self.n_tot = 0
        self.n_detect_tot = 0
        if model == 'a3c':
            #self.state_dim = (2, 10)
            self.state_dim = (7, 1)
            self.act_dim = 2
            self.pointnet = Net(self.state_dim[0], self.act_dim, device)
            if phase == 'train':
                self.pointnet.train()
            else:
                self.pointnet.load_state_dict(torch.load('models/A3C_s7_a2.pth'))
                self.pointnet.eval()
            
    def boxes_to_state(self, boxes):
        '''state = np.zeros(self.state_dim)
        state[0, :] = self.last_detect / (self.base_interval)
        state[1, :] = (self.vid_len - self.count_frames) / self.vid_len
        state[2, :] = (self.budget - self.detect_frames) / self.vid_len if self.budget > self.detect_frames else 0
        for i in range(min(len(boxes), self.state_dim[1])):
            #state[3, i] = boxes[i][4]
            #state[4, i] = boxes[i][7]
            #state[5, i] = boxes[i][11]
            state[3, i] = boxes[i][13]
            state[4, i] = boxes[i][16]
            #state[8, i] = boxes[i][18]
            #state[9, i] = boxes[i][19]'''
        state = np.zeros(self.state_dim)
        state[0] = self.last_detect / (self.base_interval)
        #state[1] = 1 #if self.budget > self.detect_frames else 0
        #state[1] = (self.vid_len - self.count_frames) / self.vid_len
        #state[2] = (self.budget - self.detect_frames) / self.vid_len if self.budget > self.detect_frames else 0
        state[1] = np.mean([b[13] for b in boxes])  if len(boxes) > 0 else 1
        state[2] = np.mean([b[16] for b in boxes])  if len(boxes) > 0 else 0
        state[3] = np.min([b[13] for b in boxes])  if len(boxes) > 0 else 1
        state[4] = np.min([b[16] for b in boxes])  if len(boxes) > 0 else 0
        state[5] = np.mean([b[4] for b in boxes])  if len(boxes) > 0 else 1
        state[6] = np.mean([abs(b[19]) for b in boxes])  if len(boxes) > 0 else 0
        return state.flatten()
        
    def reset(self, base_interval=None):
        if base_interval is not None:
            self.base_interval = base_interval
        self.budget = 60 / self.base_interval
        self.interval = self.base_interval
        self.last_detect = self.base_interval
        self.count_frames = 0
        self.act_frames = 0
            
    def update_interval(self, boxes, a=None):
        if self.model == 'a3c':
            if a is None:
                state = self.boxes_to_state(boxes)
                logits, _ = self.pointnet.forward(state)
                prob = F.softmax(logits, dim=1).data.cpu().numpy()
                a = np.argmax(prob)
                #a = prob[0,1] >= 0.1
                #print(a, prob[0, 1], state)#.reshape(self.state_dim)[:,0])
            #if a == 1:
            #    self.interval = 1
        elif self.model == 'heuristic':
            score = np.mean([b[13] for b in boxes]) if len(boxes) > 0 else 1
            self.interval = min(self.interval, self.base_interval*score)
           
    def step(self):
        act = self.interval == 1
        detect = self.last_detect >= self.interval
        self.n_tot += 1
        self.n_detect_tot += detect
        self.last_detect = 1 if detect else self.last_detect + 1
        if self.model == 'a3c':
            self.interval = self.base_interval
        elif self.model == 'heuristic':
            self.interval = self.base_interval if detect else self.interval
        self.count_frames += 1
        self.act_frames += act
        return detect

    def print_avg_interval(self):
        avg_interval = self.n_tot / (self.n_detect_tot) if self.n_detect_tot > 0 else 0
        print('Avg detection interval =', avg_interval)
        return avg_interval
        
        

class VOD(nn.Module):
    def __init__(self, phase, detector='yolov3', tracker='kcf', scheduler='fixed', detect_interval=1, n_classes=80):
        super(VOD, self).__init__()
        if phase == 'train':
            self.device = torch.device("cpu")
        else:
            #self.device = torch.device("cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iou_thres = 0.5
        self.conf_thres = 0.4
        self.detector = Detector(self.device, detector, n_classes=n_classes)
        self.tracker = tracker
        self.scheduler = Scheduler(self.device, phase, scheduler, detect_interval)
        self.n_classes = n_classes
        self.filtered = True if self.n_classes == 80 else False

        class_ids = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
                     'n01503061', 'n02924116', 'n02958343', 'n02402425',
                     'n02084071', 'n02121808', 'n02503517', 'n02118333',
                     'n02510455', 'n02342885', 'n02374451', 'n02129165',
                     'n01674464', 'n02484322', 'n03790512', 'n02324045',
                     'n02509815', 'n02411705', 'n01726692', 'n02355227',
                     'n02129604', 'n04468005', 'n01662784', 'n04530566',
                     'n02062744', 'n02391049']

        coco_classes = [4, None, 21, 1,
                        14, 5, 2, 19,
                        16, 15, 20, None,
                        None, None, 17, None,
                        None, None, 3, None,
                        None, 18, None, None,
                        None, 6, None, None,
                        None, None]        
        
        # self.class_names = load_classes("yolov3/data/coco.names")
        self.class_names = load_classes("yolov3/data/imagenet_vid.names")
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        self.valid_classes = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 21, 25])
        self.id2idx = {id: idx for idx, id in enumerate(class_ids)}
        self.coco2imgnet = {i: None for i in range(80)}
        for idx, cls in enumerate(coco_classes):
            if cls:
                self.coco2imgnet[cls] = idx
        

    def track(self):
        for i in range(len(self.trackers)):
            score, box = self.trackers[i].update(self.frame)
            x1, y1, x2, y2 = from_tracking_box(box)
            self.boxes[i][:4] = np.array([x1, y1, x2, y2])
            self.boxes[i][13] = score
            avg, var, n = self.boxes[i][14], self.boxes[i][15], self.boxes[i][17]
            self.boxes[i][14] = (avg*n + score) / (n+1)
            self.boxes[i][15] = var - ((score-avg) * (score-self.boxes[i][14]) - var) / (n+1)
            self.boxes[i][16] = (score - avg) / avg
            self.boxes[i][17] += 1
            size = (x2-x1) * (y2-y1) / (self.frame.shape[0] * self.frame.shape[1])
            self.boxes[i][19] = (size - self.boxes[i][18]) / self.boxes[i][18]
            self.boxes[i][18] = size
            
  
    def detect(self, frame):
            detections = self.detector.detect(self.frame, conf_thres=0.1)
            if self.tracker == 'none':
                self.boxes = self.update_bbox([], detections, self.filtered)
            else:
                self.boxes = self.update_bbox(self.boxes, detections, self.filtered)
                        

    def update_bbox(self, boxes, detections, filtered):
        # ========== box ==========
        # 0:4    bbox
        # 4      conf
        # 5      conf_avg
        # 6      conf_var
        # 7      conf_diff
        # 8      cls_prob
        # 9      cls_prob_avg
        # 10     cls_prob_var
        # 11     cls_prob_diff
        # 12     n_detect
        # 13     track_score
        # 14     track_score_avg
        # 15     track_score_var
        # 16     track_score_diff
        # 17     n_track
        # 18     size
        # 19     size_diff
        # 20:-1  cls_probs
        # -1     cls_pred
        # =========================
        output = []
        new_trackers = []
        associated = set()
        for i, b1 in enumerate(detections):
            match = None
            redundant = False
            conf = b1[4]
            conf_avg = conf
            conf_var = 0
            conf_diff = 0
            cls_prob = np.max(b1[5:-1])
            cls_prob_avg = cls_prob
            cls_prob_var = 0
            cls_prob_diff = 0
            n_detect = 1
            for j, b2 in enumerate(boxes):
                if iou(b1, b2) > self.iou_thres:
                    if j in associated:
                        redundant = True
                    else:
                        match = j
                        redundant = False
                        associated.add(j)
                        conf_avg = (b2[5]*b2[12] + conf) / (b2[12]+1)
                        conf_var = b2[6] + ((conf-conf_avg) * (conf-b2[5]) - b2[6]) / (b2[12]+1)
                        conf_diff = (conf - b2[5]) / b2[5]
                        cls_prob_avg = (b2[9]*b2[12] + cls_prob) / (b2[12]+1)
                        cls_prob_var = b2[10] + ((cls_prob-cls_prob_avg) * (cls_prob-b2[9]) - b2[10]) / (b2[12]+1)
                        cls_prob_diff = (cls_prob - b2[9]) / b2[9]
                        n_detect = b2[12] + 1
                        b1[:4] = (b1[:4]*conf + b2[:4]*b2[4]) / (conf + b2[4])
                        b1[5:-1] = (b1[5:-1]*conf + b2[20:-1]*b2[5]*b2[12]) / (conf + b2[5]*b2[12])
                        b1[4] = max(conf, b2[4])
                    break
            if b1[4] > self.conf_thres and not redundant:
                size = (b1[2]-b1[0]) * (b1[3]-b1[1]) / (self.frame.shape[0] * self.frame.shape[1])
                info = [conf_avg, conf_var, conf_diff, 
                        cls_prob, cls_prob_avg, cls_prob_var, cls_prob_diff, 
                        n_detect, 1, 1, 0, 0, 0, size, 0]
                b1 = np.insert(b1, 5, np.array(info))
                cls = np.argmax(b1[20:-1])
                cls = self.coco2imgnet[cls] if self.n_classes == 80 else cls
                if (not self.filtered) or (cls in self.valid_classes):
                    b1[-1] = cls
                    output.append(b1)
                    #if match is not None:
                    #    new_trackers.append(self.trackers[match])
                    if True: #else:
                        if self.tracker == 'kcf':
                            new_trackers.append(KCFTracker(False, True, True))
                            new_trackers[-1].init(to_tracking_box(b1[:4]), self.frame)
                        elif self.tracker == 'siamfc':
                            new_trackers.append(TrackerSiamFC(net_path='trackers/siamfc.pth'))
                            new_trackers[-1].init(self.frame, to_tracking_box(b1[:4]))
        self.trackers = new_trackers
        return output
    
    
    def inference(self, input_vid, output_vid):
        vid = cv2.VideoCapture(input_vid)    
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_size)

        frame_i = 0
        inference_time = 0
        self.boxes = []
        self.trackers = []
        self.scheduler.reset()

        while True:
            _, raw_frame = vid.read()    
            if raw_frame is None:
                break
            else:
                self.frame = cv2.resize(raw_frame,  (320, 320))

            prev_time = time.time()
            
            # Tracking
            if self.tracker != 'none':
                self.track()
            
            # Scheduling
            self.scheduler.update_interval(self.boxes)
            detect = self.scheduler.step()

            # Detection
            if detect:
                self.detect(self.frame)

            # Inference time
            frame_i += 1
            inference_time += time.time() - prev_time

            # Write frame to output video
            for b in self.boxes:
                x1, y1, x2, y2, conf = b[:5]
                x1, y1, x2, y2 = rescale_bbox([x1, y1, x2, y2], (320, 320), (video_size[1], video_size[0]))
                label = '%s %.2f %.2f' % (self.class_names[int(b[-1])], conf, b[13])
                plot_one_box([x1, y1, x2, y2], raw_frame, label=label, color=self.colors[int(b[-1])])
            if detect:
                plot_one_box([3, 3, video_size[0]-3, video_size[1]-3], raw_frame, label='', color=[0,0,255])
            out.write(raw_frame)
            
        print('avg fps =', frame_i / inference_time)
        self.scheduler.print_avg_interval()
        
        
    def validate(self, data_dir, val_list, output_file):
        with open(val_list, 'r') as f:
            img_list = [data_dir + v.split(' ')[0] + '.JPEG' for v in f.read().splitlines()]
    
        tp_list = []
        conf_list = []
        cls_pred_list = []
        cls_gt_list = []
        output = open(output_file, 'w')
        confusion = [[0]*31 for i in range(30)]

        for img_id in img_list:
            self.frame = cv2.imread(img_id)
            #self.frame = cv2.resize(cv2.imread(img_id), (416, 416))
            frame_i = int(img_id.split('/')[-1].split('.')[0])
            label = img_id.replace('Data', 'Annotations').replace('.JPEG', '.xml')
            if self.filtered:
                boxes_gt = parse_bbox(label, self.id2idx, self.valid_classes)
            else:
                boxes_gt = parse_bbox(label, self.id2idx, None)
            
            if frame_i == 0:
                self.boxes = []
                self.trackers = []
                self.scheduler.reset()

            # Tracking
            if self.tracker != 'none':
                self.track()
            
            # Scheduling
            self.scheduler.update_interval(self.boxes)
            detect = self.scheduler.step()

            # Detection
            if detect:
                self.detect(self.frame)

            # Evaluation
            update_confusion(confusion, self.boxes, boxes_gt, self.iou_thres)
            tp_list += get_tp(self.boxes, boxes_gt, self.iou_thres)
            conf_list += [b[4] for b in self.boxes]
            cls_pred_list += [b[-1] for b in self.boxes]
            cls_gt_list += [b[-1] for b in boxes_gt]
                
        classes, mAP = compute_mAP(tp_list, conf_list, cls_pred_list, cls_gt_list)
        np.set_printoptions(precision=1)
        print([self.class_names[c] for c in classes])
        print(np.array(mAP)*100)
        print(np.array(mAP).mean())
        avg_interval = self.scheduler.print_avg_interval()
        
        for i, c in enumerate(classes):
            output.write('{}: {}\n'.format(self.class_names[c], mAP[i]*100))
        output.write('mAP = {}\n'.format(np.array(mAP).mean()))
        for c in confusion:
            output.write(', '.join([str(cc) for cc in c]) + '\n')
        output.write('avg interval = {}'.format(avg_interval))
        output.close()
        
        
    def greedy(self, data_dir, val_list):
        with open(val_list, 'r') as f:
            img_list = [data_dir + v.split(' ')[0] + '.JPEG' for v in f.read().splitlines()]
        
        idx = 0
        vid_list = []
        for i, img_id in enumerate(img_list):
            frame_i = int(img_id.split('/')[-1].split('.')[0])
            if (frame_i == 0 and i != 0) or (i == len(img_list)-1):
                vid_list += [img_list[idx: i]]
                idx = i

        tp_list_all = []
        conf_list_all = []
        cls_pred_list_all = []
        cls_gt_list_all = []
        n_total = 0
        n_detect = 0
                
        for vid in vid_list:
            mAP_max = 0
            detect_frames = set(range(0, len(vid), self.scheduler.base_interval))
            n_total += len(vid)
            n_detect += len(detect_frames)

            for it in range(5):
                tp_list = []
                conf_list = []
                cls_pred_list = []
                cls_gt_list = []
                self.boxes = []
                self.trackers = []
                
                max_track_score = 0
                min_track_score = 1
                max_k = 0
                min_k = 0
                if it > 0 and old_k in detect_frames:
                    detect_frames.remove(old_k)
                    detect_frames.add(new_k)

                '''k = random.randint(0, len(detect_frames)-1)
                old_k = detect_frames[k]
                if len(detect_frames) == 1:
                    new_k = random.randint(0, len(vid)-1)
                elif k == 0:
                    new_k = random.randint(0, detect_frames[k+1]-1)
                elif k == len(detect_frames)-1:
                    new_k = random.randint(detect_frames[k-1]+1, len(vid)-1)
                else:
                    new_k = random.randint(detect_frames[k-1]+1, detect_frames[k+1]-1)
                detect_frames[k] = new_k'''

                for frame_i, img_id in enumerate(vid):
                    self.frame = cv2.imread(img_id)
                    label = img_id.replace('Data', 'Annotations').replace('.JPEG', '.xml')
                    boxes_gt = parse_bbox(label, self.id2idx, self.valid_classes)

                    # Tracking
                    if self.tracker != 'none':
                        self.track()

                    detect = frame_i in detect_frames
                    
                    track_score = np.mean([b[13] for b in self.boxes]) if len(self.boxes) > 0 else 1
                    if track_score > max_track_score and detect:
                        max_track_score = track_score
                        max_k = frame_i
                    if track_score < min_track_score and not detect:
                        min_track_score = track_score
                        min_k = frame_i
                        
                    # Detection
                    if detect:
                        self.detect(self.frame)

                    # Evaluation
                    tp_list += get_tp(self.boxes, boxes_gt, self.iou_thres)
                    conf_list += [b[4] for b in self.boxes]
                    cls_pred_list += [b[-1] for b in self.boxes]
                    cls_gt_list += [b[-1] for b in boxes_gt]

                _, mAP = compute_mAP(tp_list, conf_list, cls_pred_list, cls_gt_list)
                mAP = np.array(mAP).mean() if len(mAP) > 0 else 0
                if mAP >= mAP_max:
                    mAP_max = mAP
                    tp_list_best = tp_list[:]
                    conf_list_best = conf_list[:]
                    cls_pred_list_best = cls_pred_list[:]
                    cls_gt_list_best = cls_gt_list[:]
                elif it > 0 and new_k in detect_frames:
                    detect_frames.remove(new_k)
                    detect_frames.add(old_k)
                    
                new_k = min_k
                old_k = max_k
                    
            tp_list_all += tp_list_best
            conf_list_all += conf_list_best
            cls_pred_list_all += cls_pred_list_best
            cls_gt_list_all += cls_gt_list_best
            _, mAP = compute_mAP(tp_list_all, conf_list_all, cls_pred_list_all, cls_gt_list_all)
            print(np.array(mAP).mean())
            print('avg interval =', n_total / n_detect)
        
        
    def train_init(self, data_dir, train_list):
        with open(train_list, 'r') as f:
            self.vid_list = [data_dir + v.split(' ')[0] for v in f.read().splitlines()]


    def reset(self, base_interval):
        while True:
            vid = random.choice(self.vid_list)
            img_list = sorted([vid + '/' + f for f in os.listdir(vid) if f[-5:] == '.JPEG'])
            if len(img_list) <= 61:
                continue
            start = random.choice(range(len(img_list)-61))
            self.img_list = img_list[start: start+60]
            self.frame_i = 0
            self.frame = cv2.imread(self.img_list[self.frame_i])
            self.scheduler.reset(base_interval)
            self.boxes = []
            self.iou_prev = 0
            s, _, _, _ = self.step(1)
            if len(self.boxes) > 0:
                break
        return s
    
                        
    def get_reward(self, boxes):
        label = self.img_list[self.frame_i].replace('Data', 'Annotations').replace('.JPEG', '.xml')
        boxes_gt = parse_bbox(label, self.id2idx, self.valid_classes) 
        tp, tp_iou = get_tp_iou(boxes, boxes_gt, self.iou_thres)
        # r = tp_iou / len(boxes) if len(boxes) > 0 else 0 #/ (len(boxes) + len(boxes_gt) - tp + 1e-9)
        iou_diff = tp_iou - self.iou_prev
        self.iou_prev = tp_iou
        return iou_diff


    def step(self, a):        
        self.scheduler.update_interval(self.boxes, a)
        detect = self.scheduler.step()
    
        if detect:
            self.detect(self.frame)
                    
        iou_diff = self.get_reward(self.boxes)
        r = iou_diff
        
        '''if a and ((gt_diff == 0 and tp_diff > 0) or tp_diff * gt_diff > 0):                       # successful detection
            r = self.scheduler.base_interval
        elif a and ((gt_diff == 0 and tp_diff <= 0) or  tp_diff * gt_diff < 0):                    # wrong detection
            r = -1
        elif (not detect) and ((gt_diff == 0 and tp_diff < 0) or  tp_diff * gt_diff < 0):       # miss detection
            r = -self.scheduler.base_interval
        else:
            r = 0'''
        
        done = self.frame_i >= len(self.img_list)-2
        if done and self.scheduler.act_frames == 0:
            r = -10
        elif self.scheduler.act_frames > 2*self.scheduler.budget:
            done = True
            r = -10
        
        self.frame_i += 1
        self.frame = cv2.imread(self.img_list[self.frame_i])
        self.track()
        s_ = self.scheduler.boxes_to_state(self.boxes)
        return s_, r, detect, done
    

    
if __name__ == "__main__":
    
    phase = 'val'
    detector = 'ctnet'   # yolov3/ ctnet/ mb2-ssd
    tracker = 'siamfc'        # kcf/ siamfc/ none
    scheduler = 'fixed'     # fixed/ heuristic/ a3c
    detect_interval = 7
    n_classes = 30
    vod = VOD(phase, detector, tracker, scheduler, detect_interval, n_classes)
    print(detector, tracker, scheduler, detect_interval, n_classes)
    
    # Validation
    '''data_dir = '/tmp5/hank/ILSVRC2015/Data/VID/val/'
    input_file = '/tmp5/hank/ILSVRC2015/ImageSets/VID/val.txt'
    output_file = 'output/{}_{}_{}_{}_{}classes.txt'.format(detector, tracker, scheduler, detect_interval, n_classes)
    #vod.validate(data_dir, input_file, output_file)
    vod.greedy(data_dir, input_file)
    print(detector, tracker, scheduler, detect_interval, n_classes)'''
    
    
    # Inference
    data_dir = "/tmp5/hank/ILSVRC2015/Data/VID/snippets/"
    videos = ['tiger2']
    #videos = ['street', 'horse', 'tiger', 'bicycle', 'men', 'bear', 'cars', 'airplanes']
    #data_dir = "videos/"
    #videos = ['golf', 'cars']

    for vid in videos:
        if vid == 'street':
            input_vid = data_dir + "test/ILSVRC2015_test_00145000.mp4"
        elif vid == 'horse':
            input_vid = data_dir + "val/ILSVRC2015_val_00133005.mp4"
            #input_vid = data_dir + "test/ILSVRC2015_test_00254002.mp4"
        elif vid == 'tiger2':
            input_vid = data_dir + "test/ILSVRC2015_test_00021002.mp4"
            #input_vid = data_dir + "val/ILSVRC2015_val_00149003.mp4"
        elif vid == 'bicycle':
            input_vid = data_dir + "val/ILSVRC2015_val_00041008.mp4"
            #input_vid = data_dir + "val/ILSVRC2015_val_00124001.mp4"
        elif vid == 'men':
            input_vid = data_dir + "val/ILSVRC2015_val_00073000.mp4"
        elif vid == 'bear':
            input_vid = data_dir + "val/ILSVRC2015_val_00161000.mp4"
        elif vid == 'cars':
            input_vid = data_dir + "val/ILSVRC2015_val_00107000.mp4"
            #input_vid = data_dir + "cars.mp4"
        elif vid == 'airplanes':
            input_vid = data_dir + 'val/ILSVRC2015_val_00007036.mp4'
            #input_vid = data_dir + "train/ILSVRC2015_VID_train_0003/ILSVRC2015_train_01000001.mp4"
        elif vid == 'golf':
            input_vid = data_dir + "golf.mp4"
        elif vid == 'cat':
            input_vid = data_dir + 'val/ILSVRC2015_val_00037002.mp4'
        elif vid == 'bus':
            input_vid = data_dir + 'val/ILSVRC2015_val_00023001.mp4'

        # output_vid = 'videos/{}_{}.avi'.format(vid, detector)
        output_vid = 'videos/{}_{}_{}_{}_{}.avi'.format(vid, detector, tracker, scheduler, detect_interval)
        vod.inference(input_vid, output_vid)
    
