import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import kcftracker
from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
from my_utils import *
from utils_A3C import *
from pointnet import *
from VOD import *

os.environ["OMP_NUM_THREADS"] = "4"

GAMMA = 0.9
MAX_EP = 30000
UPDATE_GLOBAL_ITER = 100

N_S = 20
N_A = 2


'''class VIDEnv(nn.Module):
    def __init__(self, data_dir, train_list):
        super(VIDEnv, self).__init__()
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = Darknet("yolov3/cfg/yolov3-spp.cfg", img_size=416).to(self.device)
        load_darknet_weights(self.detector, "yolov3/weights/weights/yolov3-spp.weights")
        self.detector.eval()
        self.iou_thres = 0.5
        self.conf_thres = 0.5
        self.n_boxes = 10
        self.max_interval = 16
        
        with open(train_list, 'r') as f:
            self.vid_list = [data_dir + v.split(' ')[0] for v in f.read().splitlines()]

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
        
        self.valid_classes = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 21, 25])
        self.id2idx = {id: idx for idx, id in enumerate(class_ids)}
        self.coco2imgnet = {i: None for i in range(80)}
        for idx, cls in enumerate(coco_classes):
            if cls:
                self.coco2imgnet[cls] = idx
        

    def detect(self, conf_thres=0.1):
        img, *_ = letterbox(self.img, new_shape=416)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img/255.0).unsqueeze(0).to(self.device)
        boxes_new = []
        with torch.no_grad():
            pred, _ = self.detector(img)
            detections = my_nms_mergecls(pred, conf_thres, self.iou_thres)[0]
            if detections is not None:   
                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], self.img.shape).round()
                boxes_new = [det.data.cpu().numpy() for det in detections]
        return boxes_new


    def track(self):
        self.img_i += 1
        self.img = cv2.imread(self.img_list[self.img_i])
        for i in range(len(self.trackers)):
            score, box = self.trackers[i].update(self.img)
            x1, y1, x2, y2 = from_tracking_box(box)
            self.boxes[i][:4] = np.array([x1, y1, x2, y2])
            self.boxes[i][13] = score
            avg, var, n = self.boxes[i][14], self.boxes[i][15], self.boxes[i][17]
            self.boxes[i][14] = (avg*n + score) / (n+1)
            self.boxes[i][15] = var - ((score-avg) * (score-self.boxes[i][14]) - var) / (n+1)
            self.boxes[i][16] = (score - avg) / avg
            self.boxes[i][17] += 1
            size = (x2-x1) * (y2-y1) / (self.img.shape[0] * self.img.shape[1])
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
                size = (b1[2]-b1[0]) * (b1[3]-b1[1]) / (self.img.shape[0] * self.img.shape[1])
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


    def get_state(self):
        state = np.zeros((N_S, self.n_boxes))
        state[0, :] = self.last_detect
        state[1, :] = np.mean(self.detect_frames)
        for i in range(min(len(self.boxes), self.n_boxes)):
            state[2:, i] = self.boxes[i][4:20]
        return state.flatten()
    
    
    def get_tp(self, boxes_gt, boxes_pred):
        tp = 0
        tp_iou = 0
        matched_gt = []
        for box1 in boxes_pred:
            for i, box2 in enumerate(boxes_gt):
                if i not in matched_gt:
                    if box1[-1] == box2[-1] and iou(box1, box2) >= self.iou_thres:
                        tp += 1
                        tp_iou += iou(box1, box2)
                        matched_gt.append(i)
                        break
        return tp, tp_iou
    
                        
    def get_reward(self):
        label = self.img_list[self.img_i].replace('Data', 'Annotations').replace('.JPEG', '.xml')
        boxes_gt = parse_bbox(label, self.id2idx)
        #detections = self.detect()
        #boxes_baseline = self.update_bbox([], detections, True)        
        #tp, tp_iou = self.get_tp(boxes_baseline, boxes_gt)
        tp2, tp_iou2 = self.get_tp(self.boxes, boxes_gt)
        reward = tp_iou2 #- tp_iou
        cost = np.mean(self.detect_frames)
        #union = len(boxes_gt) + len(self.boxes) - tp
        #new_value = (tp_iou - (len(boxes_gt)-tp) - (len(self.boxes)-tp)) / (union + 1e-9)
        #reward = new_value - self.value 
        #self.value = new_value
        return reward * (1-cost)


    def reset(self):
        vid = random.choice(self.vid_list)
        self.img_list = sorted([vid + '/' + f for f in os.listdir(vid) if f[-5:] == '.JPEG'])
        self.img_i = random.choice(range(len(self.img_list)-3))
        self.img = cv2.imread(self.img_list[self.img_i])
        self.n_detects = 0
        self.last_detect = 0
        self.detect_interval = 8
        self.detect_frames = [0]
        
        detections = self.detect()
        self.boxes = self.update_bbox([], detections, True)
        
        self.trackers = []
        for i in range(min(len(self.boxes), self.n_boxes)):
            self.trackers.append(kcftracker.KCFTracker(False, True, True))
            self.trackers[-1].init(to_tracking_box(self.boxes[i][:4]), self.img)
            
        self.track()
        return self.get_state()


    def step(self, a):
        if a: # and self.last_detect >= self.detect_interval:
            self.n_detects += 1
            self.last_detect = 1
            self.detect_frames.append(1)
            detections = self.detect()
            self.boxes = self.update_bbox(self.boxes, detections, True)
            self.trackers = []
            for i in range(min(len(self.boxes), self.n_boxes)):
                self.trackers.append(kcftracker.KCFTracker(False, True, True))
                self.trackers[-1].init(to_tracking_box(self.boxes[i][:4]), self.img)
        else:
            self.last_detect += 1
            self.detect_frames.append(0)

        if len(self.detect_frames) > 30:
            del self.detect_frames[0]
            
        #if a == 0:
        #    self.detect_interval = 2
        #elif a == 1:
        #    self.detect_interval = max(2, self.detect_interval / 2)
        #elif a == 2:
        #    self.detect_interval = min(self.max_interval, self.detect_interval * 2)

        r = self.get_reward()
        done = self.img_i == len(self.img_list)-2
        if done:
            r -= self.n_detects / self.img_i
        
        self.track()
        s_ = self.get_state()
        return s_, r, done'''



class Net(nn.Module):
    def __init__(self, s_dim, a_dim, device):
        super(Net, self).__init__()
        self.device = device
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi = PointNetCls(d=s_dim, k=a_dim, feature_transform=False).to(device)
        self.v = PointNetCls(d=s_dim, k=1, feature_transform=False).to(device)
        '''self.pi1 = nn.Linear(s_dim, 50)
        self.pi2 = nn.Linear(50, 50)
        self.pi3 = nn.Linear(50, a_dim)
        self.v1 = nn.Linear(s_dim, 50)
        self.v2 = nn.Linear(50, 50)
        self.v3 = nn.Linear(50, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])'''
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = torch.FloatTensor(x).view(-1, self.s_dim, 10).to(self.device)
        logits, _, _ = self.pi(x)
        values, _, _ = self.v(x)
        '''pi1 = F.relu6(self.pi1(x))
        pi2 = F.relu6(self.pi2(pi1))
        logits = self.pi3(pi2)
        v1 = F.relu6(self.v1(x))
        v2 = F.relu6(self.v2(v1))
        values = self.v3(v2)'''
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


    
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A, torch.device("cpu"))
        data_dir = 'data/ILSVRC2015/Data/VID/train/'
        train_list = 'data/ILSVRC2015/ImageSets/VID/my_train.txt'
        self.env = VOD(scheduler='a3c', detect_interval=30)
        self.env.train_init(data_dir, train_list)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            if self.name == 'w0':
                print('========== ep {} =========='.format(self.g_ep.value))
            while True:
                a = self.lnet.choose_action(s)
                s_, r, done = self.env.step(a)
                if self.name == 'w0':
                    np.set_printoptions(precision=2)
                    print(s.reshape(N_S, 10)[:,0], a, np.array([r]))
                    
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


        
if __name__ == "__main__":
    gnet = Net(N_S, N_A, torch.device("cpu"))        # global network
    gnet.share_memory()                              # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            if global_ep.value % 50 == 0:
                print('ep {}: reward = {}'.format(global_ep.value, r))
                torch.save(gnet.state_dict(), 'A3C_detect.pth')
                np.save('rewards_detect', np.array(res))
        else:
            break
            
    [w.join() for w in workers]
    torch.save(gnet.state_dict(), 'A3C_detect.pth')
    np.save('rewards_detect', np.array(res))
    