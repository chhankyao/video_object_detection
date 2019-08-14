import os
import sys
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from trackers.kcftracker import *
from trackers.siamfc import *
from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

from A3C import *
from utils_A3C import *
from my_utils import *



class Detector(nn.Module):
    def __init__(self, device, model='yolov3'):
        super(Detector, self).__init__()
        self.device = device
        self.model = model
        if model == 'yolov3':
            self.detector = Darknet("yolov3/cfg/yolov3-spp.cfg", img_size=416).to(self.device)
            load_darknet_weights(self.detector, "yolov3/weights/weights/yolov3-spp.weights")
            self.detector.eval()
            
    def detect(self, img, conf_thres=0.5, nms_thres=0.5, img_size=416):
        boxes = []
        if self.model == 'yolov3':
            img_, *_ = letterbox(img, new_shape=img_size)
            img_ = img_[:, :, ::-1].transpose(2, 0, 1)
            img_ = np.ascontiguousarray(img_, dtype=np.float32)
            img_ = torch.from_numpy(img_/255.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred, _ = self.detector(img_)
                detections = my_nms(pred, conf_thres, nms_thres)[0]
                if detections is not None:   
                    detections[:, :4] = scale_coords(img_.shape[2:], detections[:, :4], img.shape).round()
                    boxes = [det.data.cpu().numpy() for det in detections]
        return boxes
    


class Scheduler(nn.Module):
    def __init__(self, device, phase, model='fixed', base_interval=10):
        super(Scheduler, self).__init__()
        self.device = device
        self.model = model
        self.base_interval = base_interval
        self.n_tot = 0
        self.n_detect_tot = 0
        if model == 'a3c':
            self.state_dim = (20,10)
            self.act_dim = 2
            self.pointnet = Net(self.state_dim[0], self.act_dim, device)
            if phase == 'train':
                self.pointnet.train()
            else:
                self.pointnet.load_state_dict(torch.load('models/A3C_detect3.pth'))
                self.pointnet.eval()
            
    def boxes_to_state(self, boxes):
        state = np.zeros(self.state_dim)
        state[0, :] = 0 #self.last_detect
        state[1, :] = 0 #self.interval - self.last_detect
        state[2, :] = 0 #self.interval
        state[3, :] = np.mean(self.detect_frames) if len(self.detect_frames) else 0
        for i in range(min(len(boxes), self.state_dim[1])):
            state[4:, i] = boxes[i][4:20]
        return state.flatten()
        
    def reset(self):
        self.frame_i = 0
        self.n_detect = 0
        self.last_detect = 1
        self.detect_frames = []
        self.interval = self.base_interval
            
    def update_interval(self, boxes, a):
        if self.model == 'a3c':
            if a is None:
                state = self.boxes_to_state(boxes)
                logits, _ = self.pointnet.forward(state)
                prob = F.softmax(logits, dim=1).data.cpu().numpy()
                a = np.argmax(prob)
            if a:
                self.interval = 1
            else:
                self.interval = self.base_interval
        elif self.model == 'heuristic':
            score = np.mean([b[13] for b in boxes]) if len(boxes) > 0 else 1
            self.interval = min(self.interval, self.base_interval*score)
           
    def step(self):
        detect = self.frame_i == 0 or self.last_detect >= self.interval
        if self.model == 'heuristic' and detect:
            self.interval = self.base_interval
        self.detect_frames.append(1*detect)
        if len(self.detect_frames) > self.base_interval:
            del self.detect_frames[0]
        self.frame_i += 1
        self.n_tot += 1
        self.n_detect += detect
        self.n_detect_tot += detect
        self.last_detect = 1 if detect else self.last_detect + 1
        return detect

    def print_stats(self):
        avg_interval = self.n_tot / self.n_detect_tot
        print('Avg detection interval =', avg_interval)
        


class VOD(nn.Module):
    def __init__(self, detector='yolov3', tracker='kcf', scheduler='fixed', detect_interval=10):
        super(VOD, self).__init__()
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iou_thres = 0.5
        self.conf_thres = 0.5
        self.detector = Detector(self.device, detector)
        self.tracker = tracker
        self.scheduler = Scheduler(self.device, 'train', scheduler, detect_interval)

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
        
        self.class_names = load_classes("yolov3/data/coco.names")
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
        for i, b1 in enumerate(detections):
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
                    del boxes[j]
                    break
            if b1[4] > self.conf_thres:
                size = (b1[2]-b1[0]) * (b1[3]-b1[1]) / (self.frame.shape[0] * self.frame.shape[1])
                info = [conf_avg, conf_var, conf_diff, 
                        cls_prob, cls_prob_avg, cls_prob_var, cls_prob_diff, 
                        n_detect, 1, 1, 0, 0, 0, size, 0]
                b1 = np.insert(b1, 5, np.array(info))
                cls = np.argmax(b1[20:-1])
                cls = self.coco2imgnet[cls] if filtered else cls  
                if (not filtered) or (cls in self.valid_classes):
                    b1[-1] = cls
                    output.append(b1)  
        return output
    
    
    def inference(self, input_vid, output_vid):
        vid             = cv2.VideoCapture(input_vid)    
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_size)

        track = True
        inference_time = 0
        self.boxes = []
        self.trackers = []
        self.scheduler.reset()

        while True:
            _, self.frame = vid.read()    
            if self.frame is None:
                break

            prev_time = time.time()
            
            # Tracking
            if track:
                self.track()
            
            # Scheduling
            self.scheduler.update_interval(self.boxes, None)
            detect = self.scheduler.step()

            # Detection
            if detect:
                detections = self.detector.detect(self.frame, conf_thres=0.1)
                if track:
                    self.boxes = self.update_bbox(self.boxes, detections, False)
                    self.trackers = []
                    for box in self.boxes:
                        if self.tracker == 'kcf':
                            self.trackers.append(KCFTracker(False, True, True))
                            self.trackers[-1].init(to_tracking_box(box[:4]), self.frame)
                        else:
                            self.trackers.append(TrackerSiamFC(net_path='siamfc.pth'))
                            self.trackers[-1].init(self.frame, to_tracking_box(box[:4]))
                else:
                    self.boxes = self.update_bbox([], detections, True)

            # Inference time
            inference_time += time.time() - prev_time

            # Write frame to output video
            for b in self.boxes:
                x1, y1, x2, y2, conf = b[:5]
                label = '%s %.2f %.2f' % (self.class_names[int(b[-1])], conf, b[13])
                plot_one_box([x1, y1, x2, y2], self.frame, label=label, color=self.colors[int(b[-1])])
            if detect:
                plot_one_box([3, 3, video_size[0]-3, video_size[1]-3], self.frame, label='', color=[0,0,255])
            out.write(self.frame)
            
        print('avg fps =', (self.scheduler.frame_i-1) / inference_time)
        self.scheduler.print_stats()
        
        
    def validate(self, data_dir, val_list):
        with open(val_list, 'r') as f:
            img_list = [data_dir + v.split(' ')[0] + '.JPEG' for v in f.read().splitlines()]
    
        track = True
        tp_list = []
        conf_list = []
        cls_pred_list = []
        cls_gt_list = []

        for img_id in img_list:
            self.frame = cv2.imread(img_id)
            frame_i = int(img_id.split('/')[-1].split('.')[0])
            label = img_id.replace('Data', 'Annotations').replace('.JPEG', '.xml')
            boxes_gt = parse_bbox(label, self.id2idx, self.valid_classes)
            
            if frame_i == 0:
                self.boxes = []
                self.trackers = []
                self.scheduler.reset()

            # Tracking
            if track:
                self.track()
            
            # Scheduling
            self.scheduler.update_interval(self.boxes, None)
            detect = self.scheduler.step()

            # Detection
            if detect:
                detections = self.detector.detect(self.frame, conf_thres=0.1)
                if track:
                    self.boxes = self.update_bbox(self.boxes, detections, True)
                    self.trackers = []
                    for box in self.boxes:
                        if self.tracker == 'kcf':
                            self.trackers.append(KCFTracker(False, True, True))
                            self.trackers[-1].init(to_tracking_box(box[:4]), self.frame)
                        else:
                            self.trackers.append(TrackerSiamFC(net_path='trackers/siamfc.pth'))
                            self.trackers[-1].init(self.frame, to_tracking_box(box[:4]))
                else:
                    self.boxes = self.update_bbox([], detections, True)

            # Evaluation
            tp_list += get_tp(self.boxes, boxes_gt, self.iou_thres)
            conf_list += [b[4] for b in self.boxes]
            cls_pred_list += [b[-1] for b in self.boxes]
            cls_gt_list += [b[-1] for b in boxes_gt]

            if frame_i == 0:
                mAP = compute_mAP(tp_list, conf_list, cls_pred_list, cls_gt_list)
                np.set_printoptions(precision=1)
                print(np.array(mAP)*100)
                self.scheduler.print_stats()
                
        mAP = compute_mAP(tp_list, conf_list, cls_pred_list, cls_gt_list)
        print(np.array(mAP).mean())
        self.scheduler.print_stats()
        
        
    def train_init(self, data_dir, train_list):
        with open(train_list, 'r') as f:
            self.vid_list = [data_dir + v.split(' ')[0] for v in f.read().splitlines()]


    def reset(self):
        vid = random.choice(self.vid_list)
        self.img_list = sorted([vid + '/' + f for f in os.listdir(vid) if f[-5:] == '.JPEG'])
        self.frame_i = random.choice(range(len(self.img_list)-3))
        self.frame = cv2.imread(self.img_list[self.frame_i])
        self.scheduler.reset()
        self.boxes = []
        self.rewards = []
        s, _, _ = self.step(1)
        if len(self.boxes) == 0:
            self.reset()
        return s
    
                        
    def get_reward(self):
        label = self.img_list[self.frame_i].replace('Data', 'Annotations').replace('.JPEG', '.xml')
        boxes_gt = parse_bbox(label, self.id2idx)
        tp, tp_iou = get_tp_iou(self.boxes, boxes_gt, self.iou_thres)
        r = tp_iou / (len(self.boxes) + len(boxes_gt) - tp + 1e-9)
        return r


    def step(self, a):
        self.scheduler.update_interval(self.boxes, a)
        detect = self.scheduler.step()
        
        r, c = 0, 0
        r1 = self.get_reward()
        if detect:
            detections = self.detector.detect(self.frame, conf_thres=0.1)
            self.boxes = self.update_bbox(self.boxes, detections, True)
            self.trackers = []
            for box in self.boxes:
                if self.tracker == 'kcf':
                    self.trackers.append(KCFTracker(False, True, True))
                    self.trackers[-1].init(to_tracking_box(box[:4]), self.frame)
                else:
                    self.trackers.append(TrackerSiamFC(net_path='trackers/siamfc.pth'))
                    self.trackers[-1].init(self.frame, to_tracking_box(box[:4]))

            r2 = self.get_reward()
            self.rewards.append(r2 - r1)
        
        done = (self.frame_i == len(self.img_list)-2) or self.scheduler.frame_i >= 100
        if done:
            interval_avg = self.scheduler.frame_i / self.scheduler.n_detect
            if interval_avg < 3 or interval_avg > self.scheduler.base_interval:
                r = 0
            else:
                r = np.mean(self.rewards)
        else:
            r = 0
        
        self.frame_i += 1
        self.frame = cv2.imread(self.img_list[self.frame_i])
        self.track()
        s_ = self.scheduler.boxes_to_state(self.boxes)
        return s_, r, done
    

    
if __name__ == "__main__":
    
    detector = 'yolov3'
    tracker = 'kcf'
    scheduler = 'a3c'
    detect_interval = 20
    vod = VOD(detector, tracker, scheduler, detect_interval)
    
    
    # Validation
    data_dir = 'data/ILSVRC2015/Data/VID/val/'
    input_file = 'data/ILSVRC2015/ImageSets/VID/my_val.txt'
    vod.validate(data_dir, input_file)
    
    
    # Inference
    '''data_dir = "data/ILSVRC2015/Data/VID/snippets/"
    videos = ['horse', 'bicycle', 'bear', 'cars']

    for vid in videos:
        if vid == 'street':
            input_vid = data_dir + "test/ILSVRC2015_test_00145000.mp4"
        elif vid == 'horse':
            input_vid = data_dir + "test/ILSVRC2015_test_00254002.mp4"
        elif vid == 'tiger':
            input_vid = data_dir + "test/ILSVRC2015_test_00021002.mp4"
        elif vid == 'bicycle':
            input_vid = data_dir + "val/ILSVRC2015_val_00124001.mp4"
        elif vid == 'men':
            input_vid = data_dir + "val/ILSVRC2015_val_00073000.mp4"
        elif vid == 'bear':
            input_vid = data_dir + "val/ILSVRC2015_val_00161000.mp4"
        elif vid == 'cars':
            input_vid = data_dir + "val/ILSVRC2015_val_00107000.mp4"
        elif vid == 'airplanes':
            input_vid = data_dir + "train/ILSVRC2015_VID_train_0003/ILSVRC2015_train_01000001.mp4"

        output_vid = 'videos/{}_{}_{}_{}_{}.avi'.format(vid, detector, tracker, scheduler, detect_interval)
        vod.inference(input_vid, output_vid)'''
    